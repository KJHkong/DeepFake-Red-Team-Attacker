"""Scoped source-only execution of existing frozen native controls.

No post-generation landmarks/pose/ratio analysis; original Workers are retained.
"""
from pathlib import Path
import json
import platform
import sys
from time import perf_counter
import traceback


def generate(job, failure):
    from modules.generator.adapters import mouth_actions,day9_actions,day10_actions
    from modules.generator.workers.liveportrait_static import generate_controlled
    version=job['parameters']['bank_version']
    bank=mouth_actions if version==mouth_actions.VERSION else day9_actions if version==day9_actions.VERSION else day10_actions if version==day10_actions.VERSION else None
    if bank is None:raise ValueError('Unknown frozen bank')
    entry=bank.resolve(job['parameters']['prompt_id'],job['prompt'])
    if entry['action'] not in {'mouth_closed_to_open','mouth_open_to_closed','eyes_open_to_closed','frontal_to_left','frontal_to_right','keep_frontal'}:
        raise ValueError('Action outside Source-only scope')
    controls=dict(entry['control_spec'],action_id=entry['action'],prompt_id=entry['prompt_id'],
        bank_version=bank.VERSION,bank_sha256=bank.bank_sha() if bank is day10_actions else bank.BANK_SHA,native_text_conditioning=False,
        pose_coordinate_convention='absolute yaw; positive=image left, negative=image right',
        generated_analyzer_executed=False,output_diagnostics_enabled=False)
    return generate_controlled(job,failure,controls,diagnose=False)


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from run import check_gpu,WorkerFailure
    job=json.loads(Path(sys.argv[1]).read_text());start=perf_counter()
    result=dict(status='failed',error_code=None,metadata=dict(python=platform.python_version(),python_executable=sys.executable))
    try:
        result['metadata'].update(check_gpu());result['metadata'].update(generate(job,WorkerFailure));result['status']='success'
    except WorkerFailure as error:result['error_code']=str(error)
    except Exception:
        result['error_code']='generation_failed';traceback.print_exc()
    result['metadata']['elapsed_ms']=(perf_counter()-start)*1000
    with Path(job['reply']).open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='success' else 1


if __name__=='__main__':raise SystemExit(main())
