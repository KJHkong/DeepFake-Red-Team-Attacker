"""JSON-only entry point, launched by the model's existing Python environment."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


class WorkerFailure(Exception):
    pass


def check_gpu():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    row = subprocess.check_output(["nvidia-smi", "--id="+gpu,
        "--query-gpu=memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    free, total, utilization = [int(part.strip()) for part in row.strip().split(",")]
    if total-free > 1000 or free < 6000 or utilization > 10:
        raise WorkerFailure("gpu_busy")
    return {"gpu_uuid": gpu, "free_memory_mib_before": free}


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    started = time.perf_counter()
    result = {"status": "failed", "error_code": None, "metadata": {
        "python": platform.python_version(), "python_executable": sys.executable}}
    try:
        if job["tool"] == "inswapper":
            from inswapper import generate
        else:
            result["metadata"].update(check_gpu())
            if job["tool"] == "instruct_pix2pix":
                from instruct_pix2pix import generate
            elif job["tool"] == "stylegan2_ada":
                from stylegan2_ada import generate
            elif job['tool']=='simswap':
                from simswap import generate
            elif job['tool']=='liveportrait_static':
                from liveportrait_static import generate
            elif job['tool']=='diffswap':
                from v2_day6_diffswap import generate
            elif job['tool']=='reface':
                from v2_day6_reface import generate
            else:
                raise WorkerFailure("generation_failed")
        result["metadata"].update(generate(job, WorkerFailure))
        result["status"] = "success"
    except WorkerFailure as exc:
        result["error_code"] = str(exc)
    except Exception:
        result["error_code"] = "generation_failed"
        traceback.print_exc()
    result["metadata"]["elapsed_ms"] = (time.perf_counter()-started)*1000
    Path(job["reply"]).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
