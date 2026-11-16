"""Official single-image retargeting primitives, fixed canonical numeric controls.

No text model, driver upload, video, pickle input, latent inversion or new training.
"""
from pathlib import Path
import sys
import json


def generate(job,failure):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from modules.generator.adapters.static_actions import controls_for
    controls=controls_for(job['parameters']['action_id'],job['prompt'])
    return generate_controlled(job,failure,controls)


def generate_controlled(job,failure,controls,*,diagnose=False):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image,ImageOps
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    repo,cache=Path(job['repo']),Path(job['checkpoint']);sys.path.insert(0,str(repo))
    original_session=ort.InferenceSession
    class CpuSession(original_session):
        def __init__(self,path_or_bytes,sess_options=None,providers=None,provider_options=None,**kwargs):
            options=sess_options or ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            super().__init__(path_or_bytes,sess_options=options,providers=['CPUExecutionProvider'],**kwargs)
    ort.InferenceSession=CpuSession
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper
    from src.utils.crop import prepare_paste_back,paste_back
    from src.utils.camera import get_rotation_matrix
    config=InferenceConfig(flag_use_half_precision=True,flag_normalize_lip=False,flag_do_torch_compile=False,
        checkpoint_F=str(cache/'liveportrait/base_models/appearance_feature_extractor.pth'),
        checkpoint_M=str(cache/'liveportrait/base_models/motion_extractor.pth'),
        checkpoint_G=str(cache/'liveportrait/base_models/spade_generator.pth'),
        checkpoint_W=str(cache/'liveportrait/base_models/warping_module.pth'),
        checkpoint_S=str(cache/'liveportrait/retargeting_models/stitching_retargeting_module.pth'))
    crop_cfg=CropConfig(insightface_root=str(cache/'insightface'),landmark_ckpt_path=str(cache/'liveportrait/landmark.onnx'))
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    wrapper=LivePortraitWrapper(config);cropper=Cropper(crop_cfg=crop_cfg,flag_force_cpu=True)
    with Image.open(job['inputs']['image']) as original:rgb=np.asarray(ImageOps.exif_transpose(original).convert('RGB'))
    crop=cropper.crop_source_image(rgb,crop_cfg)
    if crop is None:raise failure('no_face')
    with torch.no_grad():
        source=wrapper.prepare_source(crop['img_crop_256x256']);info=wrapper.get_kp_info(source)
        x_source=wrapper.transform_keypoint(info);features=wrapper.extract_feature_3d(source)
        landmarks=crop['lmk_crop'];x_new=x_source.clone()
        eyes,lips=wrapper.calc_ratio([landmarks])
        source_pose={k:float(info[k].item()) for k in ('pitch','yaw','roll')}
        if controls['lip_ratio'] is not None:
            ratio=wrapper.calc_combined_lip_ratio([controls['lip_ratio']],landmarks)
            x_new=x_new+wrapper.retarget_lip(x_source,ratio)
        if controls['eye_ratio'] is not None:
            ratio=wrapper.calc_combined_eye_ratio([[controls['eye_ratio']]],landmarks)
            x_new=x_new+wrapper.retarget_eye(x_source,ratio)
        if controls['target_yaw_degrees'] is not None:
            yaw=torch.full_like(info['yaw'],controls['target_yaw_degrees'])
            rotation=get_rotation_matrix(info['pitch'],yaw,info['roll'])
            x_new=info['scale'][...,None]*(info['kp']@rotation+info['exp'])
            x_new[:,:,:2]+=info['t'][:,None,:2]
        x_new=wrapper.stitching(x_source,x_new)
        decoded=wrapper.warp_decode(features,x_source,x_new)
        if not torch.isfinite(decoded['out']).all():raise failure('generation_failed')
        face=wrapper.parse_output(decoded['out'])[0]
        mask=prepare_paste_back(config.mask_crop,crop['M_c2o'],dsize=(rgb.shape[1],rgb.shape[0]))
        result=paste_back(face,crop['M_c2o'],rgb,mask)
        Image.fromarray(result).save(job['output'])
    torch.cuda.synchronize()
    receipt=dict(controls,source_pose_degrees=source_pose,source_eye_ratio=eyes[0].tolist(),
        source_lip_ratio=lips[0].tolist(),input_size=[rgb.shape[1],rgb.shape[0]],native_render_size=[512,512],
        source_original_canvas_preserved=True,paste_back=True,stitching=True,seed_used=False,
        method='Official single-image retargeting math from src/gradio_pipeline.py; no Gradio server or driver',
        known_limitations='fixed requested controls do not guarantee achieved action, extreme pose or identity preservation',
        torch=torch.__version__,onnxruntime=ort.__version__,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    if diagnose:
        import hashlib
        from v2_day6_editing_common import save_trace
        receipt['native_images']=[save_trace(Image.fromarray(crop['img_crop_256x256']),job,'native_input',['input']),
            save_trace(Image.fromarray(face),job,'native_output',['native_input'])]
        receipt['requested_keypoints_sha256']=hashlib.sha256(x_new.cpu().numpy().tobytes()).hexdigest()
        receipt['target_to_original_matrix']=crop['M_c2o'].tolist()
        try:
            out_lmk=cropper.calc_lmk_from_cropped_image(face)
            out_pose=wrapper.get_kp_info(wrapper.prepare_source(cv2.resize(face,(256,256))))
            out_eyes,out_lips=wrapper.calc_ratio([out_lmk]) if out_lmk is not None else (None,None)
            receipt['output_diagnostics']=dict(status='completed' if out_lmk is not None else 'landmarks_unavailable',
                eye_ratio=float(np.mean(out_eyes[0])) if out_eyes is not None else None,
                lip_ratio=float(np.mean(out_lips[0])) if out_lips is not None else None,
                yaw_degrees=float(out_pose['yaw'].item()),measurement='existing landmark and same-model motion estimators, not independent truth')
        except Exception as error:
            receipt['output_diagnostics']=dict(status='unavailable',reason=type(error).__name__)
    Path(job['output']).with_name('static_controls_receipt_1023.json').write_text(json.dumps(receipt,indent=2))
    return receipt
