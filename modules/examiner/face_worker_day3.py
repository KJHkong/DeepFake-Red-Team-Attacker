"""Standalone CPU dlib HOG probe on bounded validated bytes. No paths from requests."""
import base64
import hashlib
from io import BytesIO
import json
import platform
import sys
from time import perf_counter
import warnings


def main():
    payload=sys.stdin.buffer.read(32*1024**2+1)
    if len(payload)>32*1024**2:raise ValueError('Face IPC byte limit')
    job=json.loads(payload)
    raw=base64.b64decode(job['bytes'],validate=True)
    if not 0<len(raw)<=20*1024**2 or hashlib.sha256(raw).hexdigest()!=job['source_sha256']:
        raise ValueError('Face input binding')
    import dlib
    import numpy as np
    import PIL
    from PIL import Image, ImageOps
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(BytesIO(raw),formats=['PNG','JPEG','WEBP']) as source:
            if source.width*source.height>25_000_000 or getattr(source,'n_frames',1)!=1:
                raise ValueError('Face image safety limit')
            source.load();image=ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((640,640),Image.Resampling.LANCZOS)
    started=perf_counter()
    faces=dlib.get_frontal_face_detector()(np.asarray(image),1)
    print(json.dumps({'source_sha256':job['source_sha256'],'face_count':len(faces),
        'elapsed_ms':(perf_counter()-started)*1000,'method':'dlib_hog_frontal_maxedge640_upsample1',
        'runtime':{'python':platform.python_version(),'dlib':dlib.__version__,'numpy':np.__version__,'pillow':PIL.__version__}}))


if __name__=='__main__':main()
