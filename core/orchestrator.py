import json
from pathlib import Path
from time import perf_counter

from core.artifacts import LocalArtifactAccess, RunStore
from core.inputs import LocalImageIngestor
from modules.examiner.mock import MockExaminer
from modules.evaluator.mock import MockEvaluator
from modules.generator.registry import GeneratorRegistry
from modules.generator.catalog import build_generator_registry
from modules.modifier.mock import MockModifier
from modules.modifier.pipeline import ModifierPipeline
from modules.modifier.registry import ModifierRegistry
from modules.planner.rule import RuleBasedPlanner
from modules.reference_bank.index import EmptyReferenceBank
from modules.refiner.mock import MockRefiner
from modules.visualization.sheets import pipeline_sheet
from shared.contracts import (Examiner, Evaluator, InputIngestor, ModuleFailure, Modifier, Planner,
                              ReferenceBank, Refiner)
from shared.input_requirements import validate_requirements
from shared.schemas import (Artifact, AttackRequest, Candidate, ErrorInfo, ExaminerResult, EvaluationResult, InputPolicy,
                            ModifierConfig, ModifierOperation, ModuleResult, Plan, ReferenceQuery, RunMetadata, Status, now)


class Orchestrator:
    def run_examined(self, request: AttackRequest, *, face_detector=None):
        """Explicit internal Day3 workflow; public content_only/full_pipeline semantics stay intact.

        Core injects a fresh Run-confined gateway. The separate immutable assessment links
        the content Run to Gate1 without mutating its historical content-generation claim.
        Evaluators consume only assessment.gate.decision == passed, never Run.state.
        """
        from uuid import uuid4
        from core.gate1_day3 import decide_gate1
        from modules.examiner.per_image_day3 import PerImageExaminer
        from shared.schemas import ContentAssessment
        if request.mode!='real' or request.execution_scope!='content_only':
            raise ModuleFailure('invalid_input','Examined content requires an explicit real content_only request')
        run=self.run(request)
        if run.state!='content_generated' or not run.final_candidate_id:
            return run,None
        directory=self.store._directory(run.run_id)
        candidate=next(c for c in run.candidates if c.candidate_id==run.final_candidate_id)
        access=LocalArtifactAccess(InputPolicy(allowed_roots=[str(directory)]),output_root=directory/'assessment')
        invocation=PerImageExaminer(access,face_detector=face_detector).invoke(request,candidate,[])
        result=ExaminerResult.model_validate(invocation.data['examiner_result'])
        gate=decide_gate1(run.run_id,candidate,result)
        path=directory/'content_assessment_day3.json'
        assessment=ContentAssessment(evaluation_id=uuid4().hex,run_id=run.run_id,
            candidate_id=candidate.candidate_id,artifact_id=candidate.artifact.artifact_id,
            source_sha256=candidate.artifact.sha256,examiner=result,examiner_invocation=invocation,gate=gate,metadata_path=str(path))
        with path.open('x') as stream:stream.write(assessment.model_dump_json(indent=2)+'\n')
        path.chmod(0o600)
        return run,assessment

    def __init__(self, runs_root: Path, *, planner: Planner | None = None,
                 registry: GeneratorRegistry | None = None, modifier: Modifier | None = None,
                 examiner: Examiner | None = None, references: ReferenceBank | None = None,
                 refiner: Refiner | None = None, evaluator: Evaluator | None = None,
                 output_topic: str = "overall_interface", input_policy: InputPolicy | None = None,
                 input_ingestor: InputIngestor | None = None):
        self.store = RunStore(runs_root)
        self.output_topic = output_topic
        # Server-owned stage admission; task inputs/dimensions remain manifest/Worker responsibilities.
        admission = json.loads((Path(__file__).resolve().parents[1] / "configs/content_pipeline.json").read_text())
        self.accepted_generators = frozenset(admission["accepted_generators"])
        self.planner = planner
        self.modifier = modifier or MockModifier()
        self.registry = registry if registry is not None else build_generator_registry(modifier=self.modifier)
        self.input_ingestor = input_ingestor or LocalImageIngestor(input_policy or InputPolicy())
        self.examiner = examiner or MockExaminer()
        self.references = references or EmptyReferenceBank()
        self.refiner = refiner or MockRefiner()
        self.evaluator = evaluator or MockEvaluator()

    def run(self, request: AttackRequest) -> RunMetadata:
        stamp = now()
        content = request.mock_scenario if request.mode == "mock" else request.mode
        generator_only = request.execution_scope == "generator_only"
        content_only = request.execution_scope == "content_only"
        if generator_only or content_only:
            content = request.execution_scope
        run_id, directory = self.store.create(self.output_topic, content, timestamp=stamp)
        run = RunMetadata(run_id=run_id, request=request, mock=request.mode == "mock", started_at=stamp,
                          purpose="generator_stage" if generator_only else "content_generation" if content_only else "attack")
        if generator_only or content_only:
            run.assessment_status = dict.fromkeys(
                ["examiner", "business_distribution", "quality", "identity", "detector", "gate1", "gate2"], "not_evaluated")
        self.store.save(run)
        deadline = perf_counter() + request.budget.max_duration_s
        modifier_pipeline = None
        if content_only:
            config = ModifierConfig.model_validate_json((Path(__file__).resolve().parents[1] / "configs/modifier.json").read_text())
            # Only this Run's generated/intermediate artifacts are readable; no original/business directory access.
            access = LocalArtifactAccess(InputPolicy(allowed_roots=[str(directory)], max_file_bytes=config.max_file_bytes,
                max_pixels=config.max_pixels), output_root=directory / "artifacts/modifier")
            modifier_pipeline = ModifierPipeline(ModifierRegistry(access), config)
        planner = self.planner or RuleBasedPlanner(modifier_catalog=modifier_pipeline)

        def stage(name, function, inputs=(), *, mock=True, stop_on_error=True, step=None):
            if perf_counter() >= deadline:
                raise ModuleFailure("budget_exceeded", "Run budget exhausted between stages")
            started, stamp = perf_counter(), now()
            error, value, worker_result = None, None, None
            status = Status.SUCCESS
            try:
                value = function()
                if isinstance(value, ModuleResult):
                    worker_result = value
                    status, error = value.status, value.error
                    value = value.candidate
                if isinstance(value, Plan):
                    run.plan = value
                    status = Status.SUCCESS if value.status == "ready" else Status.UNSUPPORTED_GENERATOR if value.status == "unsupported" else Status.PLAN_FAILED
                    error = value.error
                elif hasattr(value, "status"):
                    status = value.status
                    if status != Status.SUCCESS and hasattr(value, "detector"):
                        error = value.detector.error
            except ModuleFailure as exc:
                error = ErrorInfo(code=exc.code, message=str(exc), field=exc.field)
                status = Status(exc.code) if exc.code in Status._value2member_map_ else Status.MODULE_FAILED
            except Exception:
                error = ErrorInfo(code="module_failed", message=f"{name} execution failed")
                status = Status.MODULE_FAILED
            outputs = []
            if worker_result and name == "modifier":
                # Keep completed intermediates even when the chain has no successful final Candidate.
                for raw in worker_result.data.get("completed_artifacts", []):
                    artifact = Artifact.model_validate(raw)
                    if artifact.artifact_id not in {a.artifact_id for a in run.artifacts}:
                        run.artifacts.append(artifact)
                outputs = worker_result.outputs
            if isinstance(value, Candidate):
                if not (worker_result and name == "modifier"):
                    outputs = [value.artifact.artifact_id]
                if value.artifact.artifact_id not in {a.artifact_id for a in run.artifacts}:
                    run.artifacts.append(value.artifact)
                if value.candidate_id not in {c.candidate_id for c in run.candidates}:
                    run.candidates.append(value)
            elif isinstance(value, Plan):
                outputs = ["plan"]
            elif isinstance(value, ExaminerResult):
                outputs = [f"gate1:{value.candidate_id}"]
            elif isinstance(value, EvaluationResult):
                outputs = [f"evaluation:{value.candidate_id}"]
            elif isinstance(value, list):
                if name == "input_ingestion":
                    run.artifacts.extend(value)
                    run.input_artifact_ids = [artifact.artifact_id for artifact in value]
                    outputs = run.input_artifact_ids
                else:
                    outputs = [reference.sample_id for reference in value]
            if hasattr(value, "model_dump"):
                data = value.model_dump(mode="json")
            elif isinstance(value, list):
                count_field = "input_count" if name == "input_ingestion" else "reference_count"
                data = {count_field: len(value)}
            else:
                data = {}
            if step:
                data["plan_step_id"] = step.step_id
            if worker_result:
                data[f"{name}_invocation"] = worker_result.model_dump(mode="json")
            run.stages.append(ModuleResult(module=name, status=status, mock=mock, started_at=stamp,
                elapsed_ms=(perf_counter()-started)*1000, inputs=list(inputs), outputs=outputs,
                parameters={"seed": request.seed, **({"mock_scenario": request.mock_scenario} if run.mock else {}),
                            **(step.parameters if step else {})}, data=data, error=error))
            self.store.save(run)
            if error and (stop_on_error or not isinstance(value, Plan)):
                raise ModuleFailure(error.code, error.message, field=error.field)
            return value

        try:
            if request.mode == "validate_inputs":
                manifest = self.registry.manifest(request.generator) if request.generator else None
                validate_requirements(request, manifest)
                if request.parameters or request.modifier is not None:
                    raise ModuleFailure("invalid_input", "Input-only validation does not plan backend parameters or modifiers", field="parameters" if request.parameters else "modifier")
                stage("input_ingestion", lambda: self.input_ingestor.ingest(request.inputs, directory / "inputs"), ["request.inputs"], mock=False)
                if manifest:
                    run.versions["input_contract"] = f"{manifest.name}:{manifest.version}"
                run.state = "validated"
            else:
                plan = stage("planner", lambda: planner.plan(request, self.registry), ["request", "registry"],
                             mock=run.mock, stop_on_error=False)
                if plan.status != "ready":
                    # Keep the previous explicit input-contract intake for pending real tools.
                    if request.mode == "real" and request.generator and plan.error.code == "tool_unavailable":
                        manifest = self.registry.manifest(request.generator)
                        if manifest.availability == "not_integrated":
                            validate_requirements(request, manifest)
                            stage("input_ingestion", lambda: self.input_ingestor.ingest(request.inputs, directory / "inputs"), ["request.inputs"], mock=False)
                            run.versions["input_contract"] = f"{manifest.name}:{manifest.version}"
                    raise ModuleFailure(plan.error.code, plan.error.message, field=plan.error.field)
                if request.mode == "real":
                    stage("input_ingestion", lambda: self.input_ingestor.ingest(request.inputs, directory / "inputs"), ["request.inputs"], mock=False)
                    if not (generator_only or content_only):
                        raise ModuleFailure("pipeline_not_ready", "Real Gate pipeline is not enabled; use generator_only or content_only for accepted content tools", field="execution_scope")
                    if content_only:
                        self._validate_content_recipe(plan, modifier_pipeline, request)
                    candidate = self._execute_generator_plan(run, directory, stage, deadline)
                    if content_only:
                        self._execute_content_modifiers(run, candidate, directory, stage, deadline, modifier_pipeline)
                else:
                    self._execute_mock_plan(run, directory, stage)
        except ModuleFailure as exc:
            run.state = "failed"
            run.final_candidate_id = None
            run.errors.append(ErrorInfo(code=exc.code, message=str(exc), field=exc.field))
        run.finished_at = now()
        # Even rejection and execution failure receive a sheet with N/A stages.
        started = perf_counter()
        try:
            run.verification_sheet = pipeline_sheet(run, directory, topic=self.output_topic)
            run.stages.append(ModuleResult(module="visualization", status=Status.SUCCESS, mock=run.mock,
                started_at=now(), elapsed_ms=(perf_counter()-started)*1000,
                inputs=[a.artifact_id for a in run.artifacts], outputs=[run.verification_sheet.image_path],
                data={"spec_path": run.verification_sheet.spec_path}))
        except Exception:
            run.state = "failed"
            run.final_candidate_id = None
            run.errors.append(ErrorInfo(code="visualization_failed", message="Verification sheet rendering failed"))
        self.store.save(run)
        return run

    def _execute_generator_plan(self, run, directory, stage, deadline):
        """One execution path for server-admitted image Generators, without Mock Gates."""
        plan, request = run.plan, run.request
        if (plan.mock or plan.execution_scope not in {"generator_only", "content_only"} or
                plan.execution_order != ["generator"] + (["modifier"] * len(plan.modifier_chain) if plan.execution_scope == "content_only" else []) or
                plan.tool_name not in self.accepted_generators):
            raise ModuleFailure("pipeline_not_ready", "Generator has not been admitted for stage orchestration", field="execution_scope")
        tool = self.registry.get(plan.tool_name)
        if tool.manifest.mock:
            raise ModuleFailure("plan_failed", "A real Generator implementation is required")
        if tool.manifest.local_eval_status != "passed" or tool.manifest.output_type != "image":
            raise ModuleFailure("pipeline_not_ready", "Stage orchestration requires an independently accepted image Generator", field="generator")
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise ModuleFailure("budget_exceeded", "Run budget exhausted before Generator")
        # Worker timeout consumes the remaining run budget, not a fresh full budget.
        execution_request = request.model_copy(deep=True)
        execution_request.budget.max_duration_s = remaining
        inputs = [a for a in run.artifacts if a.artifact_id in run.input_artifact_ids]
        run.versions["generator"] = f"{tool.manifest.name}:{tool.manifest.version}"
        run.versions["execution_scope"] = request.execution_scope
        def invoke():
            result = self.registry.invoke(execution_request, plan, directory / "artifacts/generator", input_artifacts=inputs)
            if result.candidate is not None:
                result.candidate.metadata["generation_task"] = {
                    "run_id":run.run_id, "plan_step_id":plan.steps[0].step_id,
                    "tool_name":tool.manifest.name, "tool_version":tool.manifest.version,
                    "checkpoint_info":tool.manifest.checkpoint_info,
                    "seed":request.seed, "parameters":plan.parameters, "prompt":request.prompt,
                    "input_artifact_ids":list(run.input_artifact_ids)}
                result.candidate.metadata.update(distribution_check="not_evaluated", quality_check="not_evaluated",
                    identity_check="not_evaluated", formal_gates="not_evaluated")
                if request.execution_scope == "content_only":
                    result.candidate.metadata["refiner_effectiveness"] = "not_evaluated"
            return result
        candidate = stage("generator", invoke,
            ["request", "plan", *run.input_artifact_ids], mock=False, step=plan.steps[0])
        if not isinstance(candidate, Candidate):
            raise ModuleFailure("invalid_generator_output", "Generator returned no valid Candidate")
        if perf_counter() >= deadline:
            raise ModuleFailure("budget_exceeded", "Generator completed after the runtime deadline")
        if request.execution_scope == "generator_only":
            run.state = "generated"
        return candidate

    @staticmethod
    def _validate_content_recipe(plan, pipeline, request):
        modifier = request.modifier
        preset, chain = None, []
        if modifier and modifier.enabled:
            preset = modifier.preset
            chain = modifier.chain or ([ModifierOperation(tool_name=modifier.tool_name, parameters=modifier.parameters)]
                                       if modifier.tool_name else [])
        operations, info = pipeline.prepare(chain=None if preset is not None else chain,
                                             preset=preset, input_type="image")
        if operations != plan.modifier_chain or info != plan.modifier_recipe:
            raise ModuleFailure("plan_failed", "Modifier recipe or validation evidence changed after planning", field="modifier")

    def _execute_content_modifiers(self, run, generated, directory, stage, deadline, pipeline):
        self._validate_content_recipe(run.plan, pipeline, run.request)
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise ModuleFailure("budget_exceeded", "Run budget exhausted before Modifier")
        preset = run.plan.modifier_recipe.get("preset")
        def invoke():
            result = pipeline.invoke(generated.artifact, directory / "artifacts/modifier",
                chain=None if preset is not None else run.plan.modifier_chain, preset=preset, max_duration_s=remaining)
            for raw, planned in zip(result.data.get("step_results", []), run.plan.steps[1:]):
                raw["data"]["plan_step_id"] = planned.step_id
            if not run.plan.modifier_chain and result.status == Status.SKIPPED:
                result.candidate = generated  # Keep the actual original Candidate and Artifact, not a duplicate stage.
            if result.candidate is not None:
                result.candidate.metadata.update(distribution_check="not_evaluated", quality_check="not_evaluated",
                    identity_check="not_evaluated", formal_gates="not_evaluated", refiner_effectiveness="not_evaluated")
            return result
        candidate = stage("modifier", invoke, [generated.artifact.artifact_id], mock=False)
        if not isinstance(candidate, Candidate):
            raise ModuleFailure("modifier_failed", "Modifier chain returned no final Candidate")
        if perf_counter() >= deadline:
            raise ModuleFailure("budget_exceeded", "Content generation completed after the runtime deadline")
        run.final_candidate_id = candidate.candidate_id
        run.state = "content_generated"
        for name, version in run.plan.modifier_recipe.get("tool_versions", {}).items():
            run.versions[f"modifier.{name}"] = version

    def _execute_mock_plan(self, run, directory, stage):
        """Execute the ordered plan; Gate results always come from their owning modules."""
        plan, request = run.plan, run.request
        if not plan.mock or not plan.steps:
            raise ModuleFailure("plan_failed", "Mock execution requires an explicit executable mock plan")
        candidate, examination, refs, refined = None, None, [], False
        inputs = [a for a in run.artifacts if a.artifact_id in run.input_artifact_ids]
        for item in plan.steps:
            if item.condition == "gate1_failed" and (examination is None or examination.gate1_passed):
                continue
            if item.condition == "refined" and not refined:
                continue
            if item.condition == "gate1_passed" and (examination is None or not examination.gate1_passed):
                continue
            if item.module == "generator":
                tool = self.registry.get(item.tool_name)
                if not tool.manifest.mock or item.tool_name != plan.tool_name:
                    raise ModuleFailure("plan_failed", "Mock plan selected an incompatible generator")
                run.versions["generator"] = f"{tool.manifest.name}:{tool.manifest.version}"
                candidate = stage("generator", lambda: self.registry.invoke(request, plan, directory / "artifacts/generator", input_artifacts=inputs),
                                  ["request", "plan", *run.input_artifact_ids], step=item)
            elif item.module == "modifier":
                tool = self.registry.get_modifier(item.tool_name)
                if not tool.manifest.mock or item.tool_name != plan.modifier_name:
                    raise ModuleFailure("plan_failed", "Mock plan selected an incompatible modifier")
                candidate = stage("modifier", lambda: tool.modify(candidate, directory / "artifacts/modifier", parameters=item.parameters),
                                  [candidate.artifact.artifact_id], step=item)
            elif item.module == "reference_bank":
                refs = stage("reference_bank", lambda: self.references.retrieve(ReferenceQuery(candidate_id=candidate.candidate_id)), [candidate.candidate_id], step=item)
                run.reference_ids = [reference.sample_id for reference in refs]
            elif item.module == "examiner":
                examination = stage("examiner", lambda: self.examiner.examine(request, candidate, refs), [candidate.candidate_id], step=item)
                run.examiner_history.append(examination)
                run.gate1 = examination.gate1_passed
            elif item.module == "refiner":
                if refined or request.budget.max_refines == 0 or examination is None or examination.gate1_passed:
                    raise ModuleFailure("plan_failed", "Refinement is not permitted by the runtime gate/budget")
                candidate = stage("refiner", lambda: self.refiner.refine(candidate, examination, refs, directory / "artifacts/refiner"), [candidate.candidate_id], step=item)
                refined = True
            elif item.module == "evaluator":
                if examination is None or not examination.gate1_passed:
                    raise ModuleFailure("plan_failed", "Evaluation requires a passed runtime Gate 1")
                def evaluate():
                    run.evaluation = self.evaluator.evaluate(request, candidate)
                    return run.evaluation
                stage("evaluator", evaluate, [candidate.candidate_id], step=item)
                run.gate2 = run.evaluation.gate2_passed
        if run.evaluation is not None:
            run.state = "accepted" if run.gate2 else "rejected"
        elif run.gate1 is False:
            run.state = "rejected"
        else:
            raise ModuleFailure("plan_failed", "Execution plan did not reach a valid terminal gate")
