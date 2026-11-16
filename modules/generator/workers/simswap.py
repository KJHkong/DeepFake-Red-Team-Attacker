"""Reuse official224 ArcFace/generator/parser/alignment/paste-back; fixed local assets."""
import os
from pathlib import Path
import sys


def generate(job,failure):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    repo,cache=Path(job['repo']),Path(job['checkpoint'])
    sys.path.insert(0,str(repo));os.environ['TORCH_HOME']=str(cache/'torch')
    # All parser initialization assets must be present: no model-zoo download fallback.
    if not (cache/'torch/hub/checkpoints/resnet18-5c106cde.pth').is_file():raise failure('generation_failed')
    original_session=ort.InferenceSession
    class CpuSession(original_session):
        def __init__(self,path_or_bytes,sess_options=None,providers=None,provider_options=None,**kwargs):
            options=sess_options or ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            super().__init__(path_or_bytes,sess_options=options,providers=['CPUExecutionProvider'],**kwargs)
    ort.InferenceSession=CpuSession
    from options.test_options import TestOptions
    from models.models import create_model
    from insightface_func.face_detect_crop_single import Face_detect_crop
    from parsing_model.model import BiSeNet
    from util.reverse2original import reverse2wholeimage
    from util.add_watermark import watermark_image
    from util.norm import SpecificNorm
    sys.argv=['simswap_worker','--crop_size','224','--name','people','--which_epoch','latest',
        '--checkpoints_dir',str(cache/'checkpoints'),'--Arc_path',str(cache/'arcface_checkpoint.tar'),
        '--output_path',str(Path(job['output']).parent),'--use_mask']
    options=TestOptions().parse(save=False)
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    model=create_model(options);model.eval()
    # Reject upstream's permissive partial-load fallback using strict tensor verification.
    expected=torch.load(cache/'checkpoints/people/latest_net_G.pth',map_location='cpu')
    state=model.netG.state_dict()
    if set(expected)!=set(state) or any(not torch.equal(state[k].cpu(),expected[k]) for k in expected):
        raise failure('generation_failed')
    del expected
    app=Face_detect_crop(name='antelope',root=str(repo/'insightface_func/models'))
    app.prepare(ctx_id=0,det_thresh=.6,det_size=(512,512),mode='None')
    images={role:cv2.imread(job['inputs'][role]) for role in ('source','target')}
    aligned={}
    for role,img in images.items():
        if img is None:raise failure('invalid_input')
        found=app.get(img,224)
        if found is None:raise failure('no_face')
        aligned[role]=found
    normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    parser=BiSeNet(n_classes=19).cuda();parser.load_state_dict(torch.load(cache/'parsing/79999_iter.pth'));parser.eval()
    with torch.inference_mode():
        src=normalize(Image.fromarray(cv2.cvtColor(aligned['source'][0][0],cv2.COLOR_BGR2RGB)))[None].cuda()
        identity=F.normalize(model.netArc(F.interpolate(src,size=(112,112))),p=2,dim=1)
        target_rgb=cv2.cvtColor(aligned['target'][0][0],cv2.COLOR_BGR2RGB)
        target=torch.from_numpy(target_rgb.transpose(2,0,1).copy()).float()[None].cuda()/255
        result=model(None,target,identity,None,True)[0]
        if not torch.isfinite(result).all():raise failure('generation_failed')
        reverse2wholeimage([target],[result],aligned['target'][1],224,images['target'],
            watermark_image(str(repo/'simswaplogo/simswaplogo.png')),job['output'],False,
            pasring_model=parser,norm=SpecificNorm(),use_mask=True)
    torch.cuda.synchronize()
    return dict(torch=torch.__version__,onnxruntime=ort.__version__,native_face_size=[224,224],paste_back=True,
        face_selection='highest SCRFD confidence per official single-image implementation',detector_size=[512,512],
        detector_threshold=.6,official_logo_preserved=True,output_encoding='PNG directly from official paste-back; differs from historical JPEG serialization',
        generator_weights_exact_match=True,seed_used=False,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
