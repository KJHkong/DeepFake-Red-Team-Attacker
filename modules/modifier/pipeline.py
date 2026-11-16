"""Ordered Modifier executor and metadata preflight, shared by standalone eval and content orchestration."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from shared.contracts import ModuleFailure
from shared.output_paths import dated_name
from shared.schemas import (Candidate, ErrorInfo, ModifierConfig, ModifierOperation, ModuleResult, Status, now)


def recipe_digest(operations, registry):
    recipe = [{'tool_name':op.tool_name, 'version':registry.manifest(op.tool_name).version,
               'parameters':registry.validate(op.tool_name, op.parameters)} for op in operations]
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def preset_is_validated(selected, registry):
    if selected.local_eval_status != 'passed' or not selected.evidence_path:
        return False
    digest = recipe_digest(selected.operations, registry)
    if selected.validated_recipe_sha256 != digest:
        return False
    # Evidence is a server-configured metadata file, never a client-supplied parameter path.
    path = Path(selected.evidence_path)
    if not path.is_absolute():
        return False
    try:
        with path.open('rb') as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            return False
        proof = json.loads(raw)
        return isinstance(proof, dict) and proof.get('passed') is True and proof.get('recipe_sha256') == digest
    except (OSError, ValueError, RecursionError):
        return False


class ModifierPipeline:
    def __init__(self, registry, config: ModifierConfig | None = None):
        self.registry = registry
        self.config = config or ModifierConfig()

    def manifests(self):
        return self.registry.manifests()

    def prepare(self, *, chain=None, preset=None, input_type="image"):
        if chain is not None and preset is not None:
            raise ModuleFailure('invalid_input', 'Specify a chain or preset, not both', field='modifier')
        selected = None
        if preset is not None:
            selected = self.config.presets.get(preset)
            if selected is None:
                raise ModuleFailure('unsupported_preset', 'Modifier preset is not registered', field='modifier.preset')
            operations = selected.operations
        else:
            operations = [ModifierOperation.model_validate(op) for op in (chain or [])]
        if len(operations) > self.config.max_operations:
            raise ModuleFailure('invalid_input', 'Modifier chain exceeds the operation limit', field='modifier.chain')
        resolved = []
        for op in operations:
            manifest = self.registry.manifest(op.tool_name)
            if manifest.kind != 'modifier' or manifest.mock or manifest.output_type != 'image' or input_type != 'image':
                raise ModuleFailure('incompatible_media', 'Image Modifier requires image input and output', field='modifier.chain')
            resolved.append(ModifierOperation(tool_name=op.tool_name,
                parameters=self.registry.validate(op.tool_name, op.parameters)))
            input_type = manifest.output_type
        if selected is not None and not preset_is_validated(selected, self.registry):
            raise ModuleFailure('preset_not_validated', 'Preset has no matching successful Local Eval evidence', field='modifier.preset')
        info = {'execution_mode':'preset' if preset is not None else 'chain' if resolved else 'none',
                'preset':preset, 'max_duration_s':self.config.max_duration_s, 'recipe_sha256':recipe_digest(resolved, self.registry),
                'tool_versions':{op.tool_name:self.registry.manifest(op.tool_name).version for op in resolved}}
        if selected is not None:
            info.update(local_eval_status=selected.local_eval_status, evidence_path=selected.evidence_path,
                        validated_recipe_sha256=selected.validated_recipe_sha256)
        return resolved, info

    def invoke(self, source, output_dir: Path, *, chain=None, preset=None, max_duration_s=None) -> ModuleResult:
        started, stamp = perf_counter(), now()
        operations, results, artifacts = [], [], []
        error, failed_index = None, None
        mode = 'preset' if preset is not None else 'chain' if chain else 'none'
        try:
            operations, _ = self.prepare(chain=chain, preset=preset, input_type=source.media_type.split('/')[0])
            failed_index = None
            if not operations:
                return ModuleResult(module='modifier', status=Status.SKIPPED, mock=source.mock,
                    started_at=stamp, elapsed_ms=(perf_counter()-started)*1000, inputs=[source.artifact_id],
                    candidate=Candidate(artifact=source, mock=source.mock, metadata={'modifier':'not_run'}),
                    data={'execution_mode':'none','step_results':[], 'completed_artifacts':[],
                          'final_output_artifact_id':source.artifact_id, 'input_validation':'not_run', 'reason':'No Modifier requested'})
            deadline = started + min(self.config.max_duration_s, max_duration_s if max_duration_s is not None else self.config.max_duration_s)
            current = source
            for index, op in enumerate(operations):
                if perf_counter() >= deadline:
                    failed_index = index
                    raise ModuleFailure('budget_exceeded', 'Modifier chain budget exhausted between operators')
                target = Path(output_dir) / dated_name('modifier', f'step_{index+1:02d}_{op.tool_name}')
                failed_index = index
                step_started, step_stamp = perf_counter(), now()
                try:
                    result = self.registry.invoke(op.tool_name, current, target, parameters=op.parameters)
                except ModuleFailure as exc:
                    result = ModuleResult(module='modifier', status=Status.MODULE_FAILED, mock=source.mock,
                        started_at=step_stamp, elapsed_ms=(perf_counter()-step_started)*1000, inputs=[current.artifact_id], parameters=op.parameters,
                        data={'operator':op.tool_name, 'output_generated':False},
                        error=ErrorInfo(code=exc.code, message=str(exc), field=exc.field))
                except Exception:
                    result = ModuleResult(module='modifier', status=Status.MODULE_FAILED, mock=source.mock,
                        started_at=step_stamp, elapsed_ms=(perf_counter()-step_started)*1000, inputs=[current.artifact_id], parameters=op.parameters,
                        data={'operator':op.tool_name, 'output_generated':False},
                        error=ErrorInfo(code='modifier_failed', message='Modifier execution failed'))
                result.data['sequence_index'] = index
                results.append(result)
                if result.error or result.status != Status.SUCCESS or result.candidate is None:
                    failed_index = index
                    error = result.error or ErrorInfo(code='modifier_failed', message='Modifier did not produce a successful result')
                    break
                current = result.candidate.artifact
                artifacts.append(current)
                failed_index = None
                if perf_counter() >= deadline:
                    failed_index = index
                    raise ModuleFailure('budget_exceeded', 'Modifier completed after the chain deadline')
        except ModuleFailure as exc:
            error = ErrorInfo(code=exc.code, message=str(exc), field=exc.field)
        except (ValidationError, TypeError, ValueError):
            error = ErrorInfo(code='invalid_input', message='Modifier chain must contain structured operations')
        candidate = None
        if error is None:
            candidate = Candidate(artifact=artifacts[-1], mock=source.mock, metadata={
                'modifier_chain':[op.tool_name for op in operations], 'preset':preset,
                'quality_check':'not_evaluated', 'detector_executed':False})
        not_run = operations[len(results):]
        status = Status(error.code) if error and error.code in Status._value2member_map_ else Status.MODULE_FAILED if error else Status.SUCCESS
        return ModuleResult(module='modifier', status=status, mock=source.mock,
            started_at=stamp, elapsed_ms=(perf_counter()-started)*1000, inputs=[source.artifact_id],
            outputs=[artifact.artifact_id for artifact in artifacts],
            parameters={'chain':[op.model_dump(mode='json') for op in operations], 'preset':preset},
            data={'execution_mode':mode, 'execution_order':[op.tool_name for op in operations],
                'step_results':[result.model_dump(mode='json') for result in results],
                'completed_artifacts':[a.model_dump(mode='json') for a in artifacts],
                'failed_step_index':failed_index, 'not_run_operators':[op.tool_name for op in not_run],
                'final_output_artifact_id':candidate.artifact.artifact_id if candidate else None},
            candidate=candidate, error=error)
