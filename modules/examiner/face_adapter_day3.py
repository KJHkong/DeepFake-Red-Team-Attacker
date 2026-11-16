"""Byte-only adapter to the existing offline CPU HOG face detector environment."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter

from shared.contracts import FaceObservation, ModuleFailure


class DlibFaceDetectorDay3:
    def __init__(self,config_path,*,runner=subprocess.run):
        self.config=json.loads(Path(config_path).read_text())
        self.python=Path(self.config['python']);self.worker=Path(__file__).with_name('face_worker_day3.py')
        if (not self.python.is_absolute() or not self.python.is_file() or
                self.config['max_edge']!=640 or self.config['upsample']!=1 or
                not 0<self.config['timeout_s']<=30 or not 1<=self.config['threads']<=4):
            raise ModuleFailure('face_backend_unavailable','Face backend configuration is unavailable or unsupported')
        self.worker_sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest()
        self.runner=runner

    def detect(self,artifact,data):
        if not isinstance(data,bytes) or not 0<len(data)<=20*1024**2 or hashlib.sha256(data).hexdigest()!=artifact.sha256:
            raise ModuleFailure('face_input_changed','Face input bytes differ from the validated Artifact')
        if hashlib.sha256(self.worker.read_bytes()).hexdigest()!=self.worker_sha256:
            raise ModuleFailure('face_backend_changed','Face worker changed after initialization')
        job=json.dumps({'source_sha256':artifact.sha256,'bytes':base64.b64encode(data).decode('ascii')}).encode()
        if len(job)>32*1024**2:raise ModuleFailure('face_input_too_large','Face IPC byte limit exceeded')
        env={k:os.environ[k] for k in ['PATH','HOME','LANG'] if k in os.environ}
        env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS=str(self.config['threads']),OPENBLAS_NUM_THREADS=str(self.config['threads']),
            MKL_NUM_THREADS=str(self.config['threads']))
        started=perf_counter()
        try:
            result=self.runner([str(self.python),'-B',str(self.worker)],input=job,capture_output=True,
                timeout=self.config['timeout_s'],env=env,check=False)
        except subprocess.TimeoutExpired:
            raise ModuleFailure('face_timeout','Face detector exceeded its bounded execution time') from None
        except OSError:
            raise ModuleFailure('face_backend_unavailable','Face detector runtime could not be started') from None
        if result.returncode!=0:raise ModuleFailure('face_worker_failed','CPU face detector did not complete')
        try:
            if len(result.stdout)>64*1024:raise ValueError()
            reply=json.loads(result.stdout)
            if (reply['source_sha256']!=artifact.sha256 or type(reply['face_count']) is not int or
                    not 0<=reply['face_count']<=10000 or not math.isfinite(reply['elapsed_ms']) or
                    not 0<=reply['elapsed_ms']<=self.config['timeout_s']*1000 or
                    reply['runtime']!=self.config['runtime'] or reply['method']!='dlib_hog_frontal_maxedge640_upsample1'):
                raise ValueError()
        except (ValueError,TypeError,KeyError):
            raise ModuleFailure('invalid_face_result','Face detector response violates its binding or runtime contract') from None
        return FaceObservation(artifact.sha256,reply['face_count'],(perf_counter()-started)*1000,
            reply['method'],reply['runtime'],mock=False,details={'worker_detection_ms':reply['elapsed_ms'],
            'worker_sha256':self.worker_sha256,'device':'cpu','identity_inference':False})
