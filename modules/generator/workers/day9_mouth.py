"""Explicit bidirectional mouth profile, reusing original native retargeting math."""
from pathlib import Path
import hashlib
import json
import platform
import sys
from time import perf_counter
import traceback


def generate(job,failure):
    from modules.generator.adapters.mouth_actions import VERSION,BANK_SHA,resolve
    from modules.generator.workers.liveportrait_static import generate_controlled
    if job['parameters']['bank_version']!=VERSION:raise ValueError('Wrong mouth profile version')
    entry=resolve(job['parameters']['prompt_id'],job['prompt'])
    controls=dict(entry['control_spec'],action_id=entry['action'],prompt_id=entry['prompt_id'],bank_version=VERSION,bank_sha256=BANK_SHA,
                  required_source_state=entry['required_source_state'],target_state=entry['target_state'],native_text_conditioning=False,
                  full_prompt_retained=True,pose_coordinate_convention='Mouth-only; yaw/eyes unchanged numerically, attained pose not guaranteed')
    receipt=generate_controlled(job,failure,controls,diagnose=True)
    for trace in receipt.get('native_images',[]):
        previous=Path(trace['artifact']['path']);current=previous.with_name(previous.stem.rsplit('_',1)[0]+'_1028.png')
        if current.exists():raise FileExistsError(current)
        previous.rename(current);trace['artifact']['path']=str(current)
    if job.get('output'):
        previous=Path(job['output']).with_name('static_controls_receipt_1023.json')
        if previous.exists():
            current=previous.with_name('liveportrait_mouth_controls_receipt_1028.json')
            with current.open('x') as handle:handle.write(json.dumps(receipt,indent=2)+'\n')
            previous.unlink()  # Only this new Run's just-emitted receipt.
    return receipt


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from run import check_gpu,WorkerFailure
    job=json.loads(Path(sys.argv[1]).read_text());started=perf_counter()
    result=dict(status='failed',error_code=None,metadata=dict(python=platform.python_version(),python_executable=sys.executable))
    try:
        result['metadata'].update(check_gpu());result['metadata'].update(generate(job,WorkerFailure));result['status']='success'
    except WorkerFailure as error:result['error_code']=str(error)
    except Exception:
        result['error_code']='generation_failed';traceback.print_exc()
    result['metadata']['elapsed_ms']=(perf_counter()-started)*1000
    with Path(job['reply']).open('x') as handle:handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='success' else 1


if __name__=='__main__':raise SystemExit(main())
