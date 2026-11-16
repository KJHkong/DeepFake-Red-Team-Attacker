"""Deterministic metadata planning. Never reads inputs or executes model code."""
from pathlib import Path
from typing import Literal

from shared.contracts import ModifierCatalog, ModuleFailure, ToolRegistry
from shared.input_requirements import validate_requirements
from shared.schemas import (AttackRequest, ErrorInfo, ModifierOperation, Plan, PlannerConfig, PlanRejection, PlanStep,
                            ToolAssessment, ToolManifest)


class RuleBasedPlanner:
    def __init__(self, config: PlannerConfig | None = None, *, modifier_catalog: ModifierCatalog | None = None):
        self.modifier_catalog = modifier_catalog
        self.config = config if config is not None else PlannerConfig.model_validate_json(
            (Path(__file__).resolve().parents[2] / "configs/planner.json").read_text())

    def _check(self, request: AttackRequest, tool: ToolManifest, registry: ToolRegistry, *,
               explicit: bool, kind: Literal["generator", "modifier"]) -> ToolAssessment:
        issues = []
        field = "generator" if kind == "generator" else "modifier.tool_name"
        def reject(code, message, location=field):
            issues.append(ErrorInfo(code=code, message=message, field=location))
        if tool.mock != (request.mode == "mock"):
            reject("mode_tool_mismatch", "Tool mock/real mode does not match the request")
        if request.taxonomy not in tool.taxonomy:
            reject("taxonomy_mismatch", "Tool taxonomy does not match the request", "taxonomy")
        if kind == "generator":
            try:
                validate_requirements(request, tool)
            except ModuleFailure as exc:
                if not any(i.code == exc.code for i in issues):
                    reject(exc.code, str(exc), exc.field)
        if not registry.is_available(tool.name):
            reject("tool_unavailable", "Tool has no ready registered implementation")
        priorities = self.config.generator_priority if kind == "generator" else self.config.modifier_priority
        priority = priorities.index(tool.name) if tool.name in priorities else None
        if priority is None and tool.default_priority is not None:
            priority = len(priorities) + tool.default_priority
        if priority is None and not explicit:
            reject("priority_not_configured", "Automatic selection requires an explicit configured priority")
        capabilities = request.constraints.required_capabilities if kind == "generator" else request.modifier.required_capabilities if request.modifier else []
        if set(capabilities) - set(tool.capabilities):
            reject("capability_mismatch", "Tool does not support all requested capabilities", "constraints.required_capabilities" if kind == "generator" else "modifier.required_capabilities")
        parameters = request.parameters if kind == "generator" else request.modifier.parameters if request.modifier else {}
        if set(parameters) - set(tool.controllable_parameters):
            reject("unsupported_parameter", "Request contains parameters not supported by the tool", "parameters" if kind == "generator" else "modifier.parameters")
        if kind == "generator" and "seed" in parameters and parameters["seed"] != request.seed:
            reject("invalid_input", "parameters.seed must agree with the request seed", "parameters.seed")
        if tool.estimated_duration_s is None:
            reject("budget_unverified", "Tool has no declared duration estimate; budget cannot be planned", "budget.max_duration_s")
        matches = []
        if kind == "generator":
            matches = sorted(set(request.constraints.preferred_strengths) & set(tool.strengths))
            matches += sorted(set(request.constraints.preferred_suitable_for) & set(tool.suitable_for))
        return ToolAssessment(tool_name=tool.name, kind=kind, priority=priority, version=tool.version,
                              availability=tool.availability, estimated_duration_s=tool.estimated_duration_s,
                              eligible=not issues, issues=issues, preference_matches=matches)

    def plan(self, request: AttackRequest, tool_registry: ToolRegistry) -> Plan:
        common = dict(taxonomy=request.taxonomy, seed=request.seed, mock=request.mode == "mock", parameters={},
                      execution_scope=request.execution_scope,
                      budget=request.budget.model_copy(deep=True), constraints=request.constraints.model_copy(deep=True))
        assessments = []
        pair_errors = []
        def fail(error):
            unsupported = {"unsupported_generator", "tool_unavailable", "mode_tool_mismatch", "capability_mismatch", "taxonomy_mismatch", "priority_not_configured"}
            unmet = list(dict.fromkeys([error.message, *[issue.message for item in assessments for issue in item.issues],
                                      *[pair.error.message for pair in pair_errors]]))
            return Plan(**common, status="unsupported" if error.code in unsupported else "failed", error=error,
                        reasons=["No executable plan selected; no fallback to another execution mode"],
                        unmet_constraints=unmet, assessments=assessments, rejected_combinations=pair_errors)
        if request.mode == "validate_inputs":
            return fail(ErrorInfo(code="planning_not_requested", message="Input validation mode does not request an execution plan", field="mode"))
        tools = sorted(tool_registry.manifests(), key=lambda tool: tool.name)
        generators = [tool for tool in tools if tool.kind == "generator" and (request.generator is None or tool.name == request.generator)]
        if not generators:
            return fail(ErrorInfo(code="unsupported_generator", message="Generator is not registered", field="generator"))
        for tool in generators:
            assessments.append(self._check(request, tool, tool_registry, explicit=request.generator is not None, kind="generator"))
        checks = {item.tool_name:item for item in assessments}
        rank = lambda tool: (-len(checks[tool.name].preference_matches),
                            checks[tool.name].priority if checks[tool.name].priority is not None else 10**9, tool.name)
        candidates = sorted([tool for tool in generators if checks[tool.name].eligible], key=rank)
        if not candidates:
            closest = [tool for tool in generators if tool.mock == common["mock"] and request.taxonomy in tool.taxonomy]
            preferred = sorted(closest or generators, key=rank)[0]
            return fail(checks[preferred.name].issues[0])

        if request.execution_scope == "content_only":
            return self._content_plan(request, candidates, checks, common, fail, assessments, pair_errors)

        modifier_request = request.modifier
        modifier_name = modifier_request.tool_name if modifier_request and modifier_request.enabled else None
        needs_modifier = bool(modifier_request and modifier_request.enabled)
        if modifier_request is None and common["mock"] and self.config.default_mock_modifier:
            modifier_name, needs_modifier = self.config.default_mock_modifier, True
        modifiers = [None]
        if needs_modifier:
            choices = [tool for tool in tools if tool.kind == "modifier" and (modifier_name is None or tool.name == modifier_name)]
            if not choices:
                return fail(ErrorInfo(code="tool_unavailable", message="Requested modifier is not registered", field="modifier.tool_name"))
            for tool in choices:
                check = self._check(request, tool, tool_registry, explicit=modifier_name is not None, kind="modifier")
                checks[tool.name] = check
                assessments.append(check)
            modifiers = sorted([tool for tool in choices if checks[tool.name].eligible], key=rank)
            if not modifiers:
                return fail(checks[sorted(choices,key=rank)[0].name].issues[0])

        selected = None
        for generator in candidates:
            for modifier in modifiers:
                estimate = generator.estimated_duration_s + (modifier.estimated_duration_s if modifier else 0) + self.config.overhead_reserve_s
                if estimate > request.budget.max_duration_s:
                    pair_errors.append(PlanRejection(generator=generator.name,modifier=modifier.name if modifier else None,
                        estimated_duration_s=estimate,error=ErrorInfo(code="budget_exceeded",
                        message="Declared tool costs plus reserve exceed the duration budget",field="budget.max_duration_s")))
                    continue
                output = modifier or generator
                if (output.output_width is not None and output.output_width < request.constraints.min_width) or (
                        output.output_height is not None and output.output_height < request.constraints.min_height):
                    pair_errors.append(PlanRejection(generator=generator.name,modifier=modifier.name if modifier else None,
                        estimated_duration_s=estimate,error=ErrorInfo(code="constraints_unsatisfied",
                        message="Declared final output dimensions do not meet minimum constraints",field="constraints")))
                    continue
                selected = generator, modifier, estimate
                break
            if selected:
                break
        if selected is None:
            return fail(pair_errors[0].error)
        generator, modifier, estimate = selected
        parameters = {**generator.default_parameters, **request.parameters}
        if "seed" in generator.controllable_parameters:
            parameters["seed"] = request.seed
        dependencies = {}
        for index, asset in enumerate(request.inputs):
            dependencies.setdefault(asset.role, []).append(f"request.inputs.{index}")
        input_refs = [f"request.inputs.{index}" for index in range(len(request.inputs))]
        if request.prompt and (generator.input_requirements.prompt_required or generator.input_requirements.accepts_prompt):
            input_refs.append("request.prompt")
        steps = [PlanStep(step_id="generate",module="generator",tool_name=generator.name,inputs=input_refs,
                          output_ref="candidate.generated",parameters=parameters)]
        last = "generate"
        if modifier:
            steps.append(PlanStep(step_id="modify",module="modifier",tool_name=modifier.name,inputs=["candidate.generated"],
                depends_on=[last],output_ref="candidate.modified",parameters={**modifier.default_parameters, **(modifier_request.parameters if modifier_request else {})}))
            last = "modify"
        steps.extend([
            PlanStep(step_id="retrieve",module="reference_bank",inputs=["candidate.current"],depends_on=[last],output_ref="references"),
            PlanStep(step_id="examine",module="examiner",inputs=["candidate.current","references"],depends_on=[last,"retrieve"],output_ref="gate1"),
        ])
        if request.budget.max_refines:
            steps.extend([
                PlanStep(step_id="refine",module="refiner",inputs=["candidate.current","gate1","references"],depends_on=["examine"],output_ref="candidate.refined",condition="gate1_failed"),
                PlanStep(step_id="reexamine",module="examiner",inputs=["candidate.current","references"],depends_on=["refine"],output_ref="gate1",condition="refined"),
            ])
        steps.append(PlanStep(step_id="evaluate",module="evaluator",inputs=["candidate.current","gate1"],
            depends_on=["reexamine" if request.budget.max_refines else "examine"],output_ref="evaluation",condition="gate1_passed"))
        deferred = ["Actual output validity, business distribution and quality remain Examiner/Evaluator responsibilities",
                    "Duration estimates are not runtime guarantees; enforce stage/worker deadlines"]
        if request.constraints.require_face:
            deferred.append("require_face: verify a valid generated face at Gate 1")
        if request.mode == "mock" and request.prompt:
            deferred.append("The geometric mock fixture does not consume the supplied prompt")
        output = modifier or generator
        for dimension in ("width", "height"):
            if getattr(output, f"output_{dimension}") is None:
                deferred.append(f"min_{dimension}: check actual output dimensions at Gate 1")
        reasons = ["Explicit generator respected" if request.generator else "Selected first feasible pair in configured Generator/Modifier priority order",
                   "Registered implementation, taxonomy, mode, input roles, capabilities and parameter names checked",
                   "Modifier requested and supported" if modifier else "No modifier requested; no modifier step planned"]
        if request.execution_scope == "generator_only":
            steps = steps[:1]
            reasons.append("execution_scope=generator_only: execute only Generator; all Gates and evaluation remain unassessed")
        if checks[generator.name].preference_matches:
            reasons.append("Matched soft preferences: " + ", ".join(checks[generator.name].preference_matches))
        if pair_errors:
            reasons.append("Higher-priority pairs rejected: " + ", ".join(error.error.code for error in pair_errors))
        common["parameters"] = parameters
        return Plan(**common,tool_name=generator.name,modifier_name=modifier.name if modifier else None,
            steps=steps,execution_order=[step.module for step in steps],input_dependencies=dependencies,
            reasons=reasons,deferred_constraints=deferred,assessments=assessments,rejected_combinations=pair_errors,
            estimated_duration_s=estimate)

    def _content_plan(self, request, candidates, checks, common, fail, assessments, pair_errors):
        """Plan only accepted content stages; never executes a tool or reads image pixels."""
        modifier = request.modifier
        chain, preset = [], None
        if modifier and modifier.enabled:
            preset = modifier.preset
            if modifier.chain is not None:
                chain = modifier.chain
            elif modifier.tool_name:
                chain = [ModifierOperation(tool_name=modifier.tool_name, parameters=modifier.parameters)]
            elif modifier.parameters or modifier.required_capabilities:
                return fail(ErrorInfo(code="invalid_input", message="Content Modifier requires an explicit tool, chain or validated preset", field="modifier"))
        if (chain or preset is not None) and self.modifier_catalog is None:
            return fail(ErrorInfo(code="unsupported_modifier", message="No server Modifier catalog is configured", field="modifier"))
        selected = None
        for generator in candidates:
            try:
                if self.modifier_catalog is not None:
                    operations, recipe = self.modifier_catalog.prepare(chain=None if preset is not None else chain, preset=preset, input_type=generator.output_type)
                    manifests = {m.name:m for m in self.modifier_catalog.manifests()}
                else:
                    operations, recipe, manifests = [], {}, {}
                modifier_cost = 0
                for index, op in enumerate(operations):
                    tool = manifests[op.tool_name]
                    if request.taxonomy not in tool.taxonomy:
                        raise ModuleFailure("taxonomy_mismatch", "Modifier does not support the requested taxonomy", field="taxonomy")
                    if modifier and set(modifier.required_capabilities)-set(tool.capabilities):
                        raise ModuleFailure("capability_mismatch", "Explicit Modifier lacks required capabilities", field="modifier.required_capabilities")
                    if tool.estimated_duration_s is None:
                        raise ModuleFailure("budget_unverified", "Modifier has no declared duration estimate", field="budget.max_duration_s")
                    modifier_cost += tool.estimated_duration_s
                    assessments.append(ToolAssessment(tool_name=tool.name, kind="modifier", eligible=True,
                        priority=index, version=tool.version, availability=tool.availability, estimated_duration_s=tool.estimated_duration_s))
                if modifier_cost > recipe.get("max_duration_s", float("inf")):
                    raise ModuleFailure("budget_exceeded", "Declared Modifier costs exceed the server chain budget", field="modifier")
                estimate = generator.estimated_duration_s + modifier_cost + self.config.overhead_reserve_s
                if estimate > request.budget.max_duration_s:
                    raise ModuleFailure("budget_exceeded", "Generator and ordered Modifier costs exceed the run budget", field="budget.max_duration_s")
                selected = generator, operations, recipe, estimate
                break
            except ModuleFailure as exc:
                error = ErrorInfo(code=exc.code, message=str(exc), field=exc.field)
                pair_errors.append(PlanRejection(generator=generator.name, estimated_duration_s=generator.estimated_duration_s, error=error))
        if selected is None:
            return fail(pair_errors[0].error)
        generator, operations, recipe, estimate = selected
        parameters = {**generator.default_parameters, **request.parameters}
        if "seed" in generator.controllable_parameters:
            parameters["seed"] = request.seed
        common["parameters"] = parameters
        dependencies = {}
        for index, asset in enumerate(request.inputs):
            dependencies.setdefault(asset.role, []).append(f"request.inputs.{index}")
        refs = [f"request.inputs.{i}" for i in range(len(request.inputs))]
        if request.prompt and (generator.input_requirements.prompt_required or generator.input_requirements.accepts_prompt):
            refs.append("request.prompt")
        steps = [PlanStep(step_id="generate", module="generator", tool_name=generator.name,
                          inputs=refs, output_ref="candidate.generated", parameters=parameters)]
        for index, operation in enumerate(operations, 1):
            previous = steps[-1]
            steps.append(PlanStep(step_id=f"modify_{index:02d}", module="modifier", tool_name=operation.tool_name,
                inputs=[previous.output_ref], depends_on=[previous.step_id], output_ref=f"candidate.modified_{index:02d}",
                parameters=operation.parameters))
        return Plan(**common, tool_name=generator.name, modifier_chain=operations, modifier_recipe=recipe,
            steps=steps, execution_order=[s.module for s in steps], input_dependencies=dependencies,
            reasons=["Explicit Generator or configured deterministic priority respected", "content_only: no downstream Gates or evaluation planned",
                     "Explicit Modifier order and normalized parameters validated" if operations else "No Modifier: retain original Generator Artifact reference",
                     "Preset evidence matches actual tool versions and parameters" if preset is not None else "No automatic Modifier selection"],
            deferred_constraints=["Business distribution, face validity, quality, identity, Refiner and Detector/Gates are not_evaluated",
                                  "Dimension constraints are not Gate-checked in content_only; observed output dimensions are recorded",
                                  "Duration estimates require runtime deadline enforcement"],
            assessments=assessments, rejected_combinations=pair_errors, estimated_duration_s=estimate)
