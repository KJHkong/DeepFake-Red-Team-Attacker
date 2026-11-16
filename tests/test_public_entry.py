"""CPU release contracts. Controlled fixtures are never model-run evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import subprocess

import pytest

import attacker_cli


def task(name='swap_01', roles=('source', 'target')):
    return {'task_id': name, 'inputs': [
        {'role': role, 'path': f'input_samples/{role}.png', 'display_allowed': False}
        for role in roles]}


def test_check_does_not_contact_service_or_start_worker(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU check must not access network or workers')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    result = attacker_cli.framework_check()
    assert result['framework_ready'] is True
    assert result['real_runtime_ready'] is False
    assert result['model_calls'] == 0
    assert result['source_filter_enabled'] is False


def test_task_routes_use_existing_contract():
    rows = attacker_cli.validate_tasks({'tasks': [task(), task('edit_01', ('source',))]})
    assert [row['branch'] for row in rows] == ['swapping', 'editing']


@pytest.mark.parametrize('value', [
    {'tasks': [task(), task()]}, {'tasks': [task('../escape')]},
    {'tasks': [task(roles=('target',))]}, {'tasks': [task('hyphen-not-supported')]},
    {'tasks': [dict(task(), prompt='force an action')]},
    {'tasks': []}, {'tasks': [], 'api_key': 'not-a-real-key'},
])
def test_rejects_ambiguous_or_overriding_batch_records(value):
    with pytest.raises((ValueError, TypeError)):
        attacker_cli.validate_tasks(value)


def test_batch_preview_does_not_import_factory_or_write(tmp_path, monkeypatch):
    path = tmp_path/'tasks.json'
    path.write_text(json.dumps({'tasks': [task()]}))
    monkeypatch.setattr(attacker_cli, 'load_factory', lambda *a: pytest.fail('Preview imported factory'))
    args = SimpleNamespace(manifest=path, execute=False, factory=None, output=tmp_path/'outputs')
    result = attacker_cli.batch(args)
    assert result['scope'] == 'role_validation_only'
    assert result['model_calls'] == 0 and not args.output.exists()


def test_existing_run_is_never_overwritten(tmp_path):
    path = tmp_path/'tasks.json'
    path.write_text(json.dumps({'tasks': [task()]}))
    out = tmp_path/'existing'
    out.mkdir()
    marker = out/'keep.txt'
    marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        attacker_cli.batch(SimpleNamespace(manifest=path, execute=True, factory='unavailable:create', output=out))
    assert marker.read_text() == 'unchanged'


def test_missing_deployment_does_not_fall_back_to_mock(tmp_path):
    with pytest.raises(RuntimeError, match='deployment_local'):
        attacker_cli.load_factory('examples.runtime_factory:create_pipeline')(tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_private_idl_requires_explicit_injection():
    from modules.source_state.nisp_rpc import NispThriftClient
    with pytest.raises(RuntimeError, match='not distributed'):
        NispThriftClient().stubs()


def test_frozen_controls_and_direction_are_preserved():
    from modules.source_state.scale import select_scale_profile
    assert [select_scale_profile('mouth_closed_to_open', level)['control_spec']['lip_ratio']
            for level in ['L1', 'L2', 'L3']] == [.2, .35, .5]
    assert select_scale_profile('mouth_open_to_closed')['control_spec']['lip_ratio'] == 0
    assert select_scale_profile('eyes_open_to_closed')['control_spec']['eye_ratio'] == 0
    assert select_scale_profile('frontal_to_left', 'L3')['control_spec']['target_yaw_degrees'] == 45
    assert select_scale_profile('frontal_to_right', 'L3')['control_spec']['target_yaw_degrees'] == -45
    with pytest.raises(ValueError):
        select_scale_profile('keep_frontal')


def test_role_schedule_is_reproducible_and_balanced():
    from core.dual_branch import balanced_batch_schedule
    from collections import Counter
    a = balanced_batch_schedule(80, 80, seed=42, limits=(80, 80))
    assert a == balanced_batch_schedule(80, 80, seed=42, limits=(80, 80))
    c = Counter(row['generator'] for row in a if row['branch'] == 'swapping')
    assert sorted(c.values()) == [26, 27, 27]


def test_local_worker_config_is_operator_owned(tmp_path, monkeypatch):
    from modules.generator.catalog import build_generator_registry
    path = tmp_path/'workers.json'
    path.write_text('{}')
    monkeypatch.setenv('ATTACKER_WORKER_CONFIG', str(path))
    registry = build_generator_registry()
    assert not registry.is_available('inswapper')


def test_framework_import_is_from_this_release():
    import core.dual_branch
    root = Path(__file__).resolve().parents[1]
    assert Path(core.dual_branch.__file__).is_relative_to(root)
    assert importlib.util.find_spec('action_liveness_docker_service') is None


def test_batch_keeps_source_review_without_generator_or_detector(tmp_path, monkeypatch):
    from PIL import Image
    from core.dual_branch import DualBranchPipeline, balanced_batch_schedule
    from modules.source_state.scale import ScaleRules, ScaleScheduler, assess_scale_actions, select_scale_profile
    from shared.schemas import InputPolicy, ModuleResult, Status, now
    image = tmp_path/'controlled_geometry.png'
    Image.new('RGB', (160, 160), 'gray').save(image)
    original = image.read_bytes()
    manifest = tmp_path/'tasks.json'
    manifest.write_text(json.dumps({'tasks': [{'task_id': 'review_01', 'inputs': [
        {'role': 'source', 'path': str(image), 'display_allowed': False}]}]}))

    def unavailable(*args):
        pytest.fail('Review Source must not execute Generator, GPU guard or Detector')

    def analyze(artifact, raw):
        return ModuleResult(module='source_analysis', status=Status.SUCCESS, mock=False, started_at=now(), elapsed_ms=0,
                            data={'source_sha256': artifact.sha256, 'observations': {
                                'face_count': {'count': 2}, 'liveportrait': {'face_count': 2},
                                'nisp': {'status': 'unavailable', 'scores': {}}}, 'measurements': {}})

    def factory(out):
        return DualBranchPipeline(out, input_policy=InputPolicy(allowed_roots=[str(tmp_path)]),
            analyzer=SimpleNamespace(analyze=analyze), rules=ScaleRules(enabled=True), face_detector=None,
            detector=SimpleNamespace(infer_one=unavailable), swapping_registry_factory=unavailable,
            gpu_guard=unavailable, source_assessor=assess_scale_actions, profile_resolver=select_scale_profile,
            scheduler_factory=ScaleScheduler,
            schedule_factory=lambda s, e: balanced_batch_schedule(s, e, seed=42, limits=(1, 1)))

    monkeypatch.setattr(attacker_cli, 'load_factory', lambda name: factory)
    out = tmp_path/'new-run'
    result = attacker_cli.batch(SimpleNamespace(manifest=manifest, execute=True, factory='fixture:create', output=out))
    assert result['statuses'] == {'Review': 1}
    row = json.loads((out/'results.jsonl').read_text())
    assert row['generated'] is None and row['detector'] is None
    assert row['gate1'] == 'not_evaluated' and image.read_bytes() == original
