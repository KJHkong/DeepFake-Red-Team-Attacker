"""CPU-only REFace inverse alignment, based on upstream video paste-back equations.

No model forward, alignment re-detection, Generator choice or Detector authority.
The recorded original-coordinate FFHQ quad is required; never guess from a bbox.
"""
import numpy as np
from PIL import Image,ImageFilter

FACE_LABELS=[1,2,3,5,6,7,9]  # upstream project_ffhq remove_mask_tar_FFHQ


def alignment_matrices(quad,alignment_size=1024):
    quad=np.asarray(quad,dtype=np.float64)
    if quad.shape!=(4,2) or not np.isfinite(quad).all() or not 1<=alignment_size<=2048:
        raise ValueError('Invalid recorded alignment geometry')
    target=quad+.5
    aligned=np.array([[0,0],[0,alignment_size],[alignment_size,alignment_size],[alignment_size,0]],dtype=np.float64)
    rows=[];values=[]
    for (x,y),(u,v) in zip(target,aligned):
        rows.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        values.extend([u,v])
    try:
        forward=np.r_[np.linalg.solve(np.asarray(rows),values),1.].reshape(3,3)
        inverse=np.linalg.inv(forward)
    except np.linalg.LinAlgError as exc:raise ValueError('Singular recorded alignment geometry') from exc
    if not np.isfinite(inverse).all() or abs(inverse[2,2])<1e-12:
        raise ValueError('Invalid inverse alignment')
    inverse/=inverse[2,2]
    projected=(forward@np.column_stack([target,np.ones(4)]).T).T
    if not np.allclose(projected[:,:2]/projected[:,2:],aligned,atol=1e-6):raise ValueError('Alignment corner roundtrip failed')
    return forward,inverse


def paste_image(target,aligned,quad,*,alignment_size=1024,mode='face_mask_feather',mask=None,feather_radius=16.):
    forward,inverse=alignment_matrices(quad,alignment_size)
    if target.width*target.height>25_000_000 or aligned.size[0]!=aligned.size[1]:raise ValueError('Invalid image dimensions')
    face=aligned.convert('RGB').resize((alignment_size,alignment_size),Image.Resampling.BILINEAR)
    if mode=='official_quad':alpha=Image.new('L',face.size,255)
    elif mode=='face_mask_feather':
        if mask is None or not np.isfinite(feather_radius) or not 0<=feather_radius<=32:
            raise ValueError('A recorded parser mask and bounded feather radius are required')
        binary=(np.isin(np.asarray(mask.convert('L')),FACE_LABELS)*255).astype('uint8')
        if not np.any(binary):raise ValueError('Recorded mask has no editable face region')
        alpha=Image.fromarray(binary).resize(face.size,Image.Resampling.NEAREST).filter(ImageFilter.GaussianBlur(feather_radius))
    else:raise ValueError('Unsupported paste-back policy')
    rgba=face.convert('RGBA');rgba.putalpha(alpha)
    warped=rgba.transform(target.size,Image.Transform.PERSPECTIVE,tuple(forward.ravel()[:8]),Image.Resampling.BILINEAR,fillcolor=(0,0,0,0))
    projected_alpha=warped.getchannel('A')
    result=Image.alpha_composite(target.convert('RGBA'),warped).convert('RGB')
    before=np.asarray(target.convert('RGB'));after=np.asarray(result);outside=np.asarray(projected_alpha)==0
    unchanged=int(np.count_nonzero(np.any(before!=after,axis=2)&outside))
    return result,projected_alpha,dict(adapter_version='reface_pasteback_v1_1026',mode=mode,
        target_to_aligned=forward.tolist(),aligned_to_target=inverse.tolist(),quad_original=np.asarray(quad).tolist(),
        quad_order='TL,BL,BR,TR',pillow_half_pixel_offset=.5,alignment_size=alignment_size,
        aligned_generation_size=list(aligned.size),output_size=list(target.size),image_resample='bilinear',
        mask_labels=FACE_LABELS if mode=='face_mask_feather' else None,feather_radius=feather_radius if mode=='face_mask_feather' else 0,
        outside_mask_changed_pixels=unchanged,mask_coverage=float(np.mean(np.asarray(projected_alpha)>0)),
        image_identity_preservation='not_evaluated',model_forward_executed=False,
        source='Upstream scripts/inference_swap_video.py crop_and_align_face + PERSPECTIVE alpha composite; optional fixed parser-mask feather extension')
