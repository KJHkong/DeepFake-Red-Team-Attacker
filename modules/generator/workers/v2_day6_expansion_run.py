"""Opt-in expansion worker entry; leaves historical run.py byte-for-byte intact."""
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
from run import check_gpu,WorkerFailure


def main():
    job=json.loads(Path(sys.argv[1]).read_text());start=perf_counter()
    result={'status':'failed','error_code':None,'metadata':{'python':platform.python_version(),'python_executable':sys.executable}}
    try:
        result['metadata'].update(check_gpu())
        if job['tool']=='blendface':from v2_day6_blendface import generate
        else:raise WorkerFailure('generation_failed')
        result['metadata'].update(generate(job,WorkerFailure));result['status']='success'
    except WorkerFailure as e:result['error_code']=str(e)
    except Exception:
        result['error_code']='generation_failed';traceback.print_exc()
    result['metadata']['elapsed_ms']=(perf_counter()-start)*1000
    Path(job['reply']).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['status']=='success' else 1

if __name__=='__main__':raise SystemExit(main())
