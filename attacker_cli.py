"""Small local CLI around existing contracts; real execution needs local backends."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import re

from core.dual_branch import DualBranchPipeline, route_inputs
from shared.schemas import InputAsset

ROOT = Path(__file__).resolve().parent


def framework_check():
    """Read metadata only. This is not a GPU, service or model health probe."""
    from modules.generator.catalog import build_generator_registry
    from modules.source_state.scale import ScaleRules, select_scale_profile
    rules = ScaleRules.model_validate_json((ROOT/'configs/source_rules.json').read_text())
    # These resolvers verify the immutable banks' content hashes as well.
    actions = ['mouth_closed_to_open', 'mouth_open_to_closed', 'eyes_open_to_closed',
               'frontal_to_left', 'frontal_to_right']
    for action in actions:
        select_scale_profile(action)
    registry = build_generator_registry()
    available = [name for name in ['inswapper', 'simswap', 'liveportrait_static']
                 if registry.is_available(name)]
    return dict(framework_ready=True, real_runtime_ready=False, model_calls=0,
                source_filter_enabled=rules.enabled, validated_actions=actions,
                configured_generator_workers=available,
                scope='CPU imports, contracts, immutable controls and metadata only; no service/model verification')


def validate_tasks(document):
    """Reuse InputAsset and the existing role router, without reading image pixels."""
    if not isinstance(document, dict) or set(document) != {'tasks'}:
        raise ValueError('Manifest must contain only tasks')
    tasks = document['tasks']
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 1000:
        raise ValueError('One to 1000 explicit tasks required')
    seen, rows = set(), []
    for row in tasks:
        if not isinstance(row, dict) or set(row) != {'task_id', 'inputs'}:
            raise ValueError('Each task contains exactly task_id and inputs; no action/config overrides')
        tid = row['task_id']
        if not isinstance(tid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_]{0,79}', tid) or tid in seen:
            raise ValueError('Task IDs must be unique bounded filenames')
        if not isinstance(row['inputs'], list):
            raise ValueError('Inputs must be an InputAsset list')
        inputs = [InputAsset.model_validate(value) for value in row['inputs']]
        rows.append(dict(task_id=tid, inputs=inputs, branch=route_inputs(inputs)))
        seen.add(tid)
    return rows


def load_factory(name):
    """Trusted local operator code only, never an HTTP/client-controlled plugin."""
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', name):
        raise ValueError('Factory syntax: local_module:create_pipeline')
    module, symbol = name.split(':')
    factory = getattr(importlib.import_module(module), symbol)
    if not callable(factory):
        raise ValueError('Deployment factory must be callable')
    return factory


def batch(args):
    if args.manifest.stat().st_size > 2*1024**2:
        raise ValueError('Manifest exceeds 2 MiB')
    tasks = validate_tasks(json.loads(args.manifest.read_text()))
    counts = dict(Counter(task['branch'] for task in tasks))
    if not args.execute:
        return dict(scope='role_validation_only', tasks=len(tasks), branches=counts,
                    model_calls=0, note='Image access, Source eligibility, models and services have not been checked.')
    if args.output is None or args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Execution requires a new output directory; existing runs are never resumed/overwritten')
    if not args.factory:
        raise ValueError('Real execution requires an explicit trusted local deployment factory')
    pipeline = load_factory(args.factory)(args.output.resolve())
    if not isinstance(pipeline, DualBranchPipeline) or pipeline.root.resolve() != args.output.resolve():
        raise ValueError('Factory must bind the existing DualBranchPipeline to the requested new output root')
    if pipeline.rules.enabled is not True:
        raise ValueError('Explicit locally validated Source rule enablement required')
    # The original pipeline owns input snapshots, Source selection, scheduling,
    # Workers, Gate1, Detector and task errors; this wrapper does not recreate them.
    prepared = [pipeline.prepare(task['inputs'], task_id=task['task_id']) for task in tasks]
    prepared = pipeline.plan_batch(prepared)
    results = []
    with (args.output/'results.jsonl').open('x') as handle:
        for task in prepared:
            row = pipeline.execute(task)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            handle.flush()
            results.append(row)
    summary = dict(tasks=len(results), branches=counts, statuses=dict(Counter(row['status'] for row in results)),
                   scope='real_execution_with_operator_backends', retries=0,
                   generated_analyzer_calls=0, attack_success=None)
    with (args.output/'summary.json').open('x') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='CPU framework/config check; never starts a model/service')
    command = commands.add_parser('batch', help='Validate roles by default; explicit execution uses original pipeline')
    command.add_argument('--manifest', type=Path, required=True)
    command.add_argument('--execute', action='store_true')
    command.add_argument('--factory', help='Trusted local Python factory, e.g. examples.runtime_factory:create_pipeline')
    command.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = framework_check() if args.command == 'check' else batch(args)
    except (OSError, ValueError, TypeError, ImportError, RuntimeError, AttributeError) as exc:
        print(json.dumps(dict(status='error', error_type=type(exc).__name__, message=str(exc),
                             fallback_to_mock=False), ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
