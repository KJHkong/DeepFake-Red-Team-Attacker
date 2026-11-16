"""Server-configured isolated CPU Worker; receives bytes, never Source paths."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter
from modules.source_state.analyzer import Measurements


class LivePortraitAnalysisBackend:
    def __init__(self, config_path, *, runner=subprocess.run):
        self.config_path=Path(config_path)
        self.config=json.loads(self.config_path.read_text()); self.runner=runner
        c=self.config
        if c['device']!='cpu' or c['dtype']!='float32' or not 0<c['timeout_s']<=30 or not 1<=c['threads']<=4:
            raise ValueError('Invalid analysis runtime limits')
        self.worker=Path(c['worker']);self.worker_sha256=hashlib.sha256(self.worker.read_bytes()).hexdigest()
        self.fingerprint=hashlib.sha256(self.config_path.read_bytes()+self.worker.read_bytes()).hexdigest()
        self.provenance=dict(fingerprint=self.fingerprint,version=c['version'],config_sha256=hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                             worker_sha256=self.worker_sha256,assets=c['assets'],device='cpu',dtype='float32')

    def analyze(self, artifact, data, hog_count):
        if not isinstance(data,bytes) or not 0<len(data)<=self.config['max_image_bytes'] or hashlib.sha256(data).hexdigest()!=artifact.sha256:
            raise ValueError('Analysis Source SHA/size mismatch')
        if hashlib.sha256(self.worker.read_bytes()).hexdigest()!=self.worker_sha256 or hashlib.sha256(self.config_path.read_bytes()).hexdigest()!=self.provenance['config_sha256']:
            raise ValueError('Analysis runtime changed')
        job=dict(config=self.config,bytes=base64.b64encode(data).decode(),source_sha256=artifact.sha256,hog_count=hog_count,fingerprint=self.fingerprint)
        env={k:os.environ[k] for k in ['PATH','HOME','LANG'] if k in os.environ}
        env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                   OMP_NUM_THREADS=str(self.config['threads']),MKL_NUM_THREADS=str(self.config['threads']),OPENBLAS_NUM_THREADS=str(self.config['threads']))
        start=perf_counter()
        try:
            reply=self.runner([self.config['python'],'-B',str(self.worker)],input=json.dumps(job).encode(),capture_output=True,
                              timeout=self.config['timeout_s'],env=env,check=False)
        except subprocess.TimeoutExpired:
            raise TimeoutError('Source analysis deadline exceeded') from None
        if reply.returncode!=0 or len(reply.stdout)>256*1024:
            raise ValueError('Source analysis runtime failed')
        row=json.loads(reply.stdout)
        if (row.get('status')!='success' or row.get('source_sha256')!=artifact.sha256 or row.get('fingerprint')!=self.fingerprint
            or row.get('generator_executed') is not False or row.get('device')!='cpu' or type(row.get('face_count')) is not int):
            raise ValueError('Analysis reply envelope failed')
        m=Measurements.model_validate({k:row[k] for k in ['yaw_degrees','pitch_degrees','roll_degrees','eye_ratios','lip_ratio']})
        if m.eye_ratios is not None and any(not 0<=x<=2 for x in m.eye_ratios):raise ValueError('Invalid eye ratios')
        if not math.isfinite(row['elapsed_ms']) or row['elapsed_ms']<0:raise ValueError('Invalid timing')
        row['adapter_elapsed_ms']=(perf_counter()-start)*1000
        row['private_worker_log']=reply.stderr.decode(errors='replace')[-8000:]
        return row
