"""Optional Source-only hook around the existing Editing executor.

The caller supplies Orchestrator.run_examined; this hook has no Generator,
Registry, Gate or fallback executor of its own. Dry-run is the default.
"""
from datetime import datetime, timezone
import hashlib
from time import perf_counter
from shared.schemas import ModuleResult, Status, ErrorInfo
from modules.source_state.rules import assess_source,select_profile


def run_source_checked_edit(request, artifact, *, action, access, analyzer, config,
                            execute, level=None, manual_label='unknown', manual_trusted=False,
                            dry_run=True, postcheck=None):
    started=perf_counter()
    result = ModuleResult(module='editing_source_prepost',status=Status.SUCCESS,mock=False,
        started_at=datetime.now(timezone.utc),elapsed_ms=0,inputs=[artifact.artifact_id],
        data=dict(generator_executed=False, dry_run=dry_run, stage='source_analysis'))
    try:
        profile=select_profile(action,level)
        if (request.mode!='real' or request.generator!='liveportrait_static' or request.execution_scope!='content_only'
            or len(request.inputs)!=1 or request.inputs[0].role!='image' or request.inputs[0].path!=artifact.path
            or request.prompt!=profile['prompt_text'] or request.parameters!=profile['parameters']):
            raise ValueError('Editing Source/profile/request mismatch')
        raw=access.read(artifact)
        if hashlib.sha256(raw).hexdigest()!=artifact.sha256:raise ValueError('Source SHA mismatch')
        analysis=analyzer.analyze(artifact,raw)
        decision=assess_source(analysis,action,level=level,config=config,manual_label=manual_label,manual_trusted=manual_trusted)
        result.data.update(analysis=analysis.model_dump(mode='json'),source_decision=decision.data,
                           selected_profile=profile,stage='planned_only' if dry_run else 'source_'+decision.data['eligibility'])
        if dry_run or not config.enabled or decision.data['eligibility']!='eligible' or decision.status!=Status.SUCCESS:
            result.elapsed_ms=(perf_counter()-started)*1000
            return result
        run,assessment=execute(request)
        candidate=next((c for c in run.candidates if c.candidate_id==run.final_candidate_id),None)
        result.data.update(stage='content_executed',generator_executed=bool(candidate and candidate.metadata.get('generator_executed')), generator_invoked=any(m.module=='generator' for m in run.stages),
                           run_id=run.run_id,run_state=run.state,gate1=assessment.gate.decision if assessment else 'not_evaluated')
        result.candidate=candidate
        # The legacy keyword is accepted for call compatibility only. Generated
        # Analyzer calls are disconnected; historical offline diagnostics remain.
    except Exception as error:
        result.status=Status.MODULE_FAILED
        result.error=ErrorInfo(code='editing_source_prepost_error',message='Source pre/post hook failed without fallback or retry')
        result.data.update(stage='error',error_type=type(error).__name__)
    result.elapsed_ms=(perf_counter()-started)*1000
    return result


def build_editing_executor(profile, source_artifact, *, runs_root, input_policy, face_detector, profile_resolver=select_profile):
    """Explicit scoped binding to Orchestrator.run_examined; no global registration.

    Source bytes must still match the analyzed Artifact at snapshot ingestion,
    before any real Worker. Creating this callable does not execute a model.
    """
    from dataclasses import replace
    from pathlib import Path
    from core.inputs import LocalImageIngestor
    from core.orchestrator import Orchestrator
    from shared.contracts import ModuleFailure
    from modules.generator.catalog import build_generator_registry
    from modules.generator.registry import GeneratorRegistry
    canonical = profile_resolver(profile['action'], profile['intensity_level'])
    if canonical != profile:
        raise ValueError('Server profile changed or is not canonical')
    base = build_generator_registry().get('liveportrait_static')
    if profile['action']=='keep_frontal':
        from modules.generator.adapters.day10_actions_adapter import KeepFrontalGenerator as Adapter,KeepFrontalParameters as Params
    elif profile['action'].startswith('mouth_'):
        from modules.generator.adapters.mouth_actions_adapter import MouthGenerator as Adapter, MouthParameters as Params
    else:
        from modules.generator.adapters.day9_actions_adapter import Day9Generator as Adapter, Day9Parameters as Params
    worker = 'source_filtered_liveportrait.py'
    root = Path(__file__).resolve().parents[1]
    manifest = base.manifest.model_copy(deep=True)
    manifest.version += '-source-five-actions-v1-1028'
    manifest.version += '-source-only-pilot'
    manifest.is_default = False
    manifest.default_parameters = Params(**profile['parameters']).model_dump()
    manifest.parameter_schema = Params.model_json_schema()
    manifest.controllable_parameters = list(manifest.default_parameters)
    manifest.evidence = dict(manifest.evidence, source_filter='explicit_opt_in_only', profile=profile)
    settings = replace(base.settings, worker=str(root/'modules/generator/workers'/worker))
    manifest.worker_hint = settings.worker
    registry = GeneratorRegistry()
    registry.register(Adapter(manifest, settings))
    ingestor = LocalImageIngestor(input_policy)
    class BoundSourceInput:
        def ingest(self, assets, destination):
            if len(assets) != 1 or assets[0].role != 'image' or assets[0].path != source_artifact.path:
                raise ModuleFailure('input_changed', 'Source path/role differs from the analyzed Artifact')
            snapshots = ingestor.ingest(assets, destination)
            if len(snapshots) != 1 or snapshots[0].sha256 != source_artifact.sha256:
                raise ModuleFailure('input_changed', 'Source bytes changed after analysis; no Generator invocation')
            return snapshots
    engine = Orchestrator(runs_root, registry=registry, input_policy=input_policy,
                          input_ingestor=BoundSourceInput(), output_topic='generator_source_checked_editing')
    # Use a bound method so callers retain the original store/Registry; no executor clone.
    if face_detector is None:
        return engine.run_examined
    from functools import partial
    return partial(engine.run_examined, face_detector=face_detector)


def run_selected_source_edit(artifact, *, access, analyzer, config, scheduler,
                             execute_factory, dry_run=True, quality_override=None):
    """Analyze Source once, choose a frozen Profile, use the existing executor.

    All dependencies/configuration are server-owned. quality_override is an
    internal injection point for controlled tests, never an HTTP input field.
    No generated-image analysis, retries, alternate source or quota backfill.
    """
    from modules.source_state.rules import assess_pilot_actions, pilot_basic_quality
    from shared.schemas import AttackRequest
    started = perf_counter()
    result = ModuleResult(module='editing_source_precheck', status=Status.SUCCESS, mock=artifact.mock,
        started_at=datetime.now(timezone.utc), elapsed_ms=0, inputs=[artifact.artifact_id],
        data=dict(generator_executed=False, generator_invoked=False, dry_run=dry_run,
                  generated_analyzer_calls=0, stage='source_analysis'))
    try:
        raw = access.read(artifact)
        if hashlib.sha256(raw).hexdigest() != artifact.sha256:
            raise ValueError('Source SHA mismatch')
        analysis = analyzer.analyze(artifact, raw)
        if analysis.data.get('source_sha256') != artifact.sha256:
            raise ValueError('Analyzer Source binding changed')
        quality = quality_override if quality_override is not None else pilot_basic_quality(raw, config)
        decision = assess_pilot_actions(analysis, config=config, quality=quality)
        result.data.update(analysis=analysis.model_dump(mode='json'), source_decision=decision.data,
                           quality=quality, stage='source_'+decision.data['eligibility'])
        if decision.data['eligibility'] != 'eligible' or decision.status != Status.SUCCESS:
            return result
        profile = scheduler.select(decision)
        if profile is None:
            result.data['source_decision'].update(eligibility='review',reasons=['duplicate_source_or_schedule_unavailable'])
            return result
        result.data.update(selected_profile=profile, selection=scheduler.last_selection)
        if dry_run:
            result.data['stage'] = 'planned_only'
            return result
        request = AttackRequest(mode='real', execution_scope='content_only', generator='liveportrait_static',
            taxonomy='Deepfake/Face_Editing', inputs=[dict(role='image',path=artifact.path,display_allowed=artifact.display_allowed)],
            prompt=profile['prompt_text'], parameters=profile['parameters'], constraints=dict(require_face=True),
            budget=dict(max_duration_s=180))
        result.data['request'] = request.model_dump(mode='json')
        run, assessment = execute_factory(profile)(request)
        candidate = next((c for c in run.candidates if c.candidate_id==run.final_candidate_id), None)
        result.data.update(stage='content_executed',run_id=run.run_id,run_state=run.state,
            generator_invoked=any(m.module=='generator' for m in run.stages),
            generator_executed=bool(candidate and candidate.metadata.get('generator_executed')),
            gate1=assessment.gate.decision if assessment else 'not_evaluated',
            assessment_ref=assessment.metadata_path if assessment else None)
        result.candidate = candidate
        if run.errors:
            result.data['execution_errors'] = [e.model_dump(mode='json') for e in run.errors]
    except Exception as error:
        result.status = Status.MODULE_FAILED
        result.error = ErrorInfo(code='editing_source_precheck_error',message='Source-only execution failed without retry or fallback')
        result.data.update(stage='error',error_type=type(error).__name__)
    finally:
        result.elapsed_ms = (perf_counter()-started)*1000
    return result
