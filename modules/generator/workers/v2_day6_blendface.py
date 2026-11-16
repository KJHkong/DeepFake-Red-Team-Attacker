"""Official BlendSwap forward + fixed InsightFace/FFHQ alignment and native alpha paste.

No model changes: forward hook retains the author's generated face and learned mask.
Full-frame uses raw generated RGB + that same alpha, rather than blending twice.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys


def generate(job,failure):
    import cv2
    import dlib
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    import torch
    from torchvision.transforms.functional import to_tensor
    base=Path(job['checkpoint']).resolve().parents[2];repo=Path(job['repo']);folder=Path(job['output']).parent
    torch.set_num_threads(4);cv2.setNumThreads(4)
    sys.path.insert(0,str(base/'representative_repos/Synthetic/Avatar_Digital_Human/LivePortrait'))
    from src.utils.dependencies.insightface.app import FaceAnalysis
    from src.utils.dependencies.insightface.utils.face_align import norm_crop
    from src.utils.dependencies.insightface.model_zoo import model_zoo
    original=model_zoo.PickableInferenceSession.__init__
    def session_init(self,model_path,**kwargs):
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;kwargs['sess_options']=options
        original(self,model_path,**kwargs)
    model_zoo.PickableInferenceSession.__init__=session_init
    face_root=base/'caches/representative_models/InSwapper/face'
    if not (face_root/'models/buffalo_l/det_10g.onnx').is_file():raise failure('generation_failed')
    app=FaceAnalysis(name='buffalo_l',root=str(face_root),allowed_modules=['detection'],providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1,det_size=(512,512),det_thresh=.5)
    source=cv2.imread(job['inputs']['source']);target=Image.open(job['inputs']['target']).convert('RGB')
    faces=app.get(source)
    if not faces:raise failure('no_face')
    face=max(faces,key=lambda f:float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
    src=Image.fromarray(cv2.cvtColor(norm_crop(source,landmark=face.kps,image_size=112),cv2.COLOR_BGR2RGB))
    align_file=base/'representative_repos/Deepfake/Face_Swapping/REFace/src/utils/alignmengt.py'
    spec=importlib.util.spec_from_file_location('fixed_ffhq_alignment',align_file);alignment=importlib.util.module_from_spec(spec);spec.loader.exec_module(alignment)
    predictor_path=base/'caches/representative_models/REFace/Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat'
    predictor=dlib.shape_predictor(str(predictor_path));detector=dlib.get_frontal_face_detector()
    if not len(detector(np.asarray(target))):raise failure('no_face')
    c,x,y=alignment.compute_transform(target,predictor,detector=detector,scale=1.)
    quad=np.stack([c-x-y,c-x+y,c+x+y,c+x-y]);aligned_target=alignment.crop_image(target,1024,quad.copy()).convert('RGB')
    target256=aligned_target.resize((256,256),Image.Resampling.BILINEAR)
    src.save(folder/'generator_blendface_source_aligned_1026.png');target256.save(folder/'generator_blendface_target_aligned_1026.png')
    sys.path.insert(0,str(repo/'swapping'))
    from blendswap import BlendSwap
    model=BlendSwap();state=torch.load(str(Path(job['checkpoint'])/'blendswap.pth'),map_location='cpu')
    model.load_state_dict(state,strict=True);del state
    model.eval().cuda();torch.cuda.reset_peak_memory_stats();trace={}
    handles=[model.G_ema.register_forward_hook(lambda m,i,o:trace.update(raw=o.detach())),
        model.mask_head_ema.register_forward_hook(lambda m,i,o:trace.update(alpha=o.detach().sigmoid()))]
    with torch.inference_mode():
        output=model(to_tensor(target256)[None].cuda(),to_tensor(src)[None].cuda())
    for h in handles:h.remove()
    if not all(torch.isfinite(t).all() for t in [output,*trace.values()]):raise failure('generation_failed')
    rgb=lambda t:Image.fromarray((t[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype('uint8'))
    aligned_output=rgb(output);raw_rgb=rgb(trace['raw']);alpha=Image.fromarray((trace['alpha'][0,0].cpu().numpy()*255).clip(0,255).astype('uint8'),'L')
    # Preserve author-aligned composite, raw model output and alpha independently.
    aligned_output.save(folder/'generator_blendface_aligned_output_1026.png');raw_rgb.save(folder/'generator_blendface_raw_face_1026.png');alpha.save(folder/'generator_blendface_learned_alpha_1026.png')
    # Same half-pixel original-coordinate homography as the audited FFHQ inverse.
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from modules.generator.adapters.v2_day6_reface_pasteback import alignment_matrices
    forward,inverse=alignment_matrices(quad,1024)
    rgba=raw_rgb.resize((1024,1024),Image.Resampling.BILINEAR).convert('RGBA')
    rgba.putalpha(alpha.resize((1024,1024),Image.Resampling.BILINEAR))
    warped=rgba.transform(target.size,Image.Transform.PERSPECTIVE,tuple(forward.ravel()[:8]),Image.Resampling.BILINEAR,fillcolor=(0,0,0,0))
    final=Image.alpha_composite(target.convert('RGBA'),warped).convert('RGB')
    outside=np.asarray(warped.getchannel('A'))==0
    changed=int(np.count_nonzero(np.any(np.asarray(final)!=np.asarray(target),axis=2)&outside))
    if changed:raise failure('generation_failed')
    final.save(job['output']);torch.cuda.synchronize()
    artifacts=[]
    for path in sorted(folder.glob('generator_blendface_*.png')):
        if str(path)==job['output']:continue
        raw=path.read_bytes();im=Image.open(path)
        artifacts.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw),width=im.width,height=im.height,media_type='image/png'))
    return dict(torch=torch.__version__,model_state_strict=True,official_repo_commit='0203d91d083eedc5b5b554f9b2380f060532587a',
        source_alignment='InsightFace ArcFace112; largest source bbox, det512/conf0.5 CPU',target_alignment='FFHQ1024 -> bilinear256; first dlib face',
        source_keypoints=face.kps.tolist(),target_quad_original=quad.tolist(),target_to_aligned=forward.tolist(),aligned_to_target=inverse.tolist(),
        fullframe_policy='warp raw G_ema RGB and exact official sigmoid mask_head_ema alpha; preserve outside alpha',outside_alpha_changed_pixels=changed,
        native_face_size=[256,256],original_target_size=list(target.size),paste_back=True,seed_used=False,
        intermediates=artifacts,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
