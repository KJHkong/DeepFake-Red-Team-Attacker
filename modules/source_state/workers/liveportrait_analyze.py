"""CPU-only byte IPC. Load detection, landmark and motion models, never synthesis."""
import base64
import contextlib
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings


def analyze(job):
    started = perf_counter()
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import yaml
    from PIL import Image, ImageOps
    config = job['config']
    raw = base64.b64decode(job['bytes'], validate=True)
    if not 0 < len(raw) <= config['max_image_bytes'] or hashlib.sha256(raw).hexdigest() != job['source_sha256']:
        raise ValueError('Invalid analysis bytes')
    if type(job['hog_count']) is not int or job['hog_count'] < 0:
        raise ValueError('Invalid independent face count')
    for asset in config['assets']:
        if hashlib.sha256(Path(asset['path']).read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError('Analysis model changed')
    repo, cache = Path(config['repo']), Path(config['checkpoint'])
    model_config = repo/'src/config/models.yaml'
    if hashlib.sha256(model_config.read_bytes()).hexdigest() != config['models_config_sha256']:
        raise ValueError('Motion configuration changed')
    sys.path.insert(0, str(repo))
    torch.set_num_threads(config['threads'])
    original_session = ort.InferenceSession
    loaded = []
    class CpuSession(original_session):
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = config['threads']; options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options=options, providers=['CPUExecutionProvider'], **kwargs)
            loaded.append(dict(model=Path(path_or_bytes).name, providers=self.get_providers()))
    ort.InferenceSession = CpuSession
    from src.config.crop_config import CropConfig
    from src.utils.cropper import Cropper
    from src.modules.motion_extractor import MotionExtractor
    from src.utils.camera import headpose_pred_to_degree
    from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
    params = yaml.safe_load(model_config.read_text())['model_params']['motion_extractor_params']
    motion = MotionExtractor(**params).cpu()
    motion.load_state_dict(torch.load(cache/'liveportrait/base_models/motion_extractor.pth', map_location='cpu'), strict=True)
    motion.eval()
    crop_cfg = CropConfig(insightface_root=str(cache/'insightface'), landmark_ckpt_path=str(cache/'liveportrait/landmark.onnx'), max_face_num=0)
    cropper = Cropper(crop_cfg=crop_cfg, flag_force_cpu=True)
    initialized_ms = (perf_counter()-started)*1000
    measured = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(BytesIO(raw), formats=['JPEG','PNG','WEBP']) as image:
            if image.width*image.height > config['max_pixels'] or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Analysis pixel/frame limit')
            image.load(); rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    faces = cropper.face_analysis_wrapper.get(np.ascontiguousarray(rgb[..., ::-1]), flag_do_landmark_2d_106=True, max_face_num=0)
    count = len(faces)
    row = dict(status='success', source_sha256=job['source_sha256'], face_count=count,
               yaw_degrees=None,pitch_degrees=None,roll_degrees=None,eye_ratios=None,lip_ratio=None,
               motion_executed=False,device='cpu',dtype='float32',loaded_onnx=loaded,
               initialization_ms=initialized_ms, generator_executed=False,
               runtime=dict(torch=torch.__version__,onnxruntime=ort.__version__), fingerprint=job['fingerprint'])
    policy=config.get('face_count_policy','agreement')
    if policy not in ('agreement','controlled_source_or'):raise ValueError('Invalid count policy')
    if policy=='controlled_source_or' and config.get('controlled_sources') is not True:
        raise ValueError('OR policy requires explicitly controlled Sources')
    single=(count==1 or job['hog_count']==1) if policy=='controlled_source_or' else count==job['hog_count']==1
    row['face_count_policy']=policy
    if single and count>0:
        crop = cropper.crop_source_image(rgb, crop_cfg)
        if crop is None:
            raise ValueError('Single-face crop failed')
        x = crop['img_crop_256x256'][None].astype(np.float32)/255.
        tensor = torch.from_numpy(np.clip(x,0,1)).permute(0,3,1,2)
        with torch.no_grad(): info = motion(tensor)
        row.update(**{k+'_degrees':float(headpose_pred_to_degree(info[k]).flatten()[0].item()) for k in ['yaw','pitch','roll']},
                   eye_ratios=calc_eye_close_ratio(crop['lmk_crop'][None])[0].tolist(),
                   lip_ratio=float(calc_lip_close_ratio(crop['lmk_crop'][None])[0][0]),motion_executed=True,
                   crop_to_original_matrix=crop['M_c2o'].tolist())
    row['elapsed_ms'] = (perf_counter()-measured)*1000
    row['worker_elapsed_ms'] = (perf_counter()-started)*1000
    return row


if __name__ == '__main__':
    try:
        payload = sys.stdin.buffer.read(32*1024**2+1)
        if len(payload) > 32*1024**2:raise ValueError('Input IPC too large')
        with contextlib.redirect_stdout(sys.stderr): result = analyze(json.loads(payload))
    except Exception as error:
        import traceback
        traceback.print_exc(file=sys.stderr)
        result = dict(status='error',error_type=type(error).__name__,generator_executed=False)
    print(json.dumps(result,allow_nan=False))
