"""Deployment binding template, not a bundled/working private service deployment."""
from importlib import import_module
from pathlib import Path

from core.dual_branch import DualBranchPipeline, balanced_batch_schedule
from modules.source_state.scale import ScaleRules, ScaleScheduler, assess_scale_actions, select_scale_profile


def create_pipeline(output_root):
    # deployment_local.py is supplied by the operator and excluded from Git.
    # It must provide real, authorized dependencies; there is no Mock fallback.
    try:
        local = import_module('deployment_local')
    except ModuleNotFoundError as exc:
        raise RuntimeError('Provide deployment_local.py with real analyzer, face_detector, detector, '
                           'input_policy, swapping_registry and gpu_guard; private services are not distributed') from exc
    root = Path(__file__).resolve().parents[1]
    rules = ScaleRules.model_validate_json((root/'configs/source_rules.local.json').read_text())
    return DualBranchPipeline(
        output_root, input_policy=local.input_policy, analyzer=local.analyzer, rules=rules,
        face_detector=local.face_detector, detector=local.detector,
        swapping_registry_factory=local.swapping_registry, gpu_guard=local.gpu_guard,
        source_assessor=assess_scale_actions, profile_resolver=select_scale_profile,
        scheduler_factory=ScaleScheduler,
        schedule_factory=lambda s, e: balanced_batch_schedule(s, e, seed=rules.seed,
                                                            limits=local.batch_limits))
