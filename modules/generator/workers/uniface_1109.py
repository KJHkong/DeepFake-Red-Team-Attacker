"""Official UniFace256 FP16 forward + explicit frozen FFHQ/parser canvas adapter.

Native model: https://github.com/xc-csc101/UniFace/blob/main/generate_swap.py
The adapter preserves the raw aligned result; it is not native UniFace paste-back.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback


def generate(job,failure):
    import cv2
    import dlib
    import numpy as np
    from PIL import Image
    import torch
    from torchvision import transforms
    import torch.nn.functional as F
    folder=Path(job['output']).parent;repo=Path(job['repo']);cache=Path(job['checkpoint'])
    support=json.loads((cache/'support_models.json').read_text());paths=support['paths']
    torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark=False
    start=perf_counter();spec=importlib.util.spec_from_file_location('uniface_fixed_alignment',paths['alignment'])
    alignment=importlib.util.module_from_spec(spec);spec.loader.exec_module(alignment)
    detector=dlib.get_frontal_face_detector();predictor=dlib.shape_predictor(paths['landmark'])
    images={};aligned={};quads={}
    for role in ('source','target'):
        with Image.open(job['inputs'][role]) as im:images[role]=im.convert('RGB')
        faces=detector(np.asarray(images[role]))
        if not len(faces):raise failure('no_face')
        c,x,y=alignment.compute_transform(images[role],predictor,detector=detector,scale=1.)
        quad=np.stack([c-x-y,c-x+y,c+x+y,c+x-y]);quads[role]=quad
        aligned[role]=alignment.crop_image(images[role],1024,quad.copy(),enable_padding=False).convert('RGB')
        aligned[role].resize((256,256),Image.Resampling.BILINEAR).save(folder/f'uniface_{role}_aligned_1109.png')
    preprocess_ms=(perf_counter()-start)*1000
    sys.path.insert(0,str(repo))
    import generate_swap as official
    start=perf_counter();checkpoint=torch.load(cache/'500000.pt',map_location='cpu')
    official.args=checkpoint['train_args']
    model=official.Model().half().cuda()
    model.g_ema.load_state_dict(checkpoint['g_ema'],strict=True)
    model.e_ema.load_state_dict(checkpoint['e_ema'],strict=True)
    training_args={k:v for k,v in vars(official.args).items() if isinstance(v,(str,int,float,bool,type(None)))}
    del checkpoint
    model.eval();load_ms=(perf_counter()-start)*1000
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5),inplace=True)])
    src=transform(aligned['source']).unsqueeze(0).half().cuda()
    tgt=transform(aligned['target']).unsqueeze(0).half().cuda()
    torch.cuda.synchronize();forward=perf_counter()
    with torch.no_grad():
        _,_,output=model([tgt,src])
    torch.cuda.synchronize();forward_ms=(perf_counter()-forward)*1000
    if tuple(output.shape)!=(1,3,256,256) or not torch.isfinite(output).all():raise failure('generation_failed')
    native=folder/'uniface_aligned_output_1109.png'
    official.save_image(output[0],str(native))
    float_path=folder/'uniface_aligned_float_1109.npy';np.save(float_path,output[0].float().cpu().numpy())
    # Original parser implementation/checkpoint, only a local pretrained backbone.
    sys.path.insert(0,str(Path(paths['parser_model']).parents[1]))
    import torch.utils.model_zoo as modelzoo
    def local_resnet(url,*args,**kwargs):
        if url!='https://download.pytorch.org/models/resnet18-5c106cde.pth':raise failure('generation_failed')
        return torch.load(paths['resnet_checkpoint'],map_location='cpu')
    modelzoo.load_url=local_resnet
    from parsing_model.model import BiSeNet
    parser=BiSeNet(n_classes=19)
    parser.load_state_dict(torch.load(paths['parser_checkpoint'],map_location='cpu'),strict=True);parser.eval()
    normalized=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    start=perf_counter()
    with torch.no_grad():
        logits=parser(normalized(aligned['target'].resize((512,512),Image.Resampling.BILINEAR)).unsqueeze(0))[0]
        labels=logits.argmax(dim=1)[0].cpu().numpy().astype('uint8')
    Image.fromarray(labels).save(folder/'uniface_target_parser_1109.png')
    selected=np.isin(labels,support['face_parser_labels'])
    if not np.any(selected):raise failure('generation_failed')
    # Reuse audited inverse transform/alpha compositor with a binary mask.
    root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root))
    from modules.generator.adapters.v2_day6_reface_pasteback import paste_image
    with Image.open(native) as im:raw_face=im.convert('RGB')
    result,alpha,geometry=paste_image(images['target'],raw_face,quads['target'],alignment_size=1024,
        mask=Image.fromarray(selected.astype('uint8')),feather_radius=support['feather_radius'])
    if geometry['outside_mask_changed_pixels']!=0 or result.size!=images['target'].size:raise failure('generation_failed')
    result.save(job['output']);alpha.save(folder/'uniface_projected_alpha_1109.png')
    geometry.update(adapter_version=support['policy_version'],source_quad_original=quads['source'].tolist(),
        mask_labels=support['face_parser_labels'],binary_mask_projection=True,
        native_model_output_size=[256,256],upstream_uniface_native_paste_back=False,
        adapter_extension='Fixed FFHQ1024 alignment, BiSeNet target-face classes, feather16 and inverse original-canvas paste')
    geo=folder/'uniface_transform_1109.json';geo.write_text(json.dumps(geometry,indent=2)+'\n');post_ms=(perf_counter()-start)*1000
    artifacts=[]
    for name in ['source_aligned','target_aligned','aligned_output','target_parser','projected_alpha']:
        path=folder/f'uniface_{name}_1109.png'
        with Image.open(path) as im:width,height=im.size;im.load()
        data=path.read_bytes();artifacts.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),media_type='image/png',width=width,height=height,size_bytes=len(data)))
    return dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        precision='official FP16 model and inputs',seed=0,strict_ema_weights_loaded=True,original_model_source_unchanged=True,
        checkpoint_train_args=training_args,model_load_ms=load_ms,preprocess_ms=preprocess_ms,forward_ms=forward_ms,
        parser_paste_ms=post_ms,paste_back=True,paste_adapter_version=support['policy_version'],upstream_native_paste_back=False,
        original_target_size=list(images['target'].size),native_face_size=[256,256],modifiers=False,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        face_selection='Existing FFHQ helper first HOG detection; no alternate detection/score-based retry',
        transform_path=str(geo),transform_sha256=hashlib.sha256(geo.read_bytes()).hexdigest(),intermediates=artifacts,
        exact_float_evidence={float_path.name:hashlib.sha256(float_path.read_bytes()).hexdigest()})


def main():
    prefix=Path(sys.executable).parent.parent
    if os.environ.get('UNIFACE_PRIVATE_ENV_READY')!='1':
        root=Path(__file__).resolve().parents[3]
        env=dict(os.environ,UNIFACE_PRIVATE_ENV_READY='1',CUDA_HOME=str(prefix),
            PATH=str(prefix/'bin')+':'+os.environ.get('PATH',''),
            CC=str(prefix/'bin/x86_64-conda-linux-gnu-gcc'),CXX=str(prefix/'bin/x86_64-conda-linux-gnu-g++'),
            LD_LIBRARY_PATH=str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            TORCH_CUDA_ARCH_LIST='8.6',MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(root/'.tool_assets/uniface/torch_extensions'))
        os.execve(sys.executable,[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]],env)
    from run import check_gpu,WorkerFailure
    job=json.loads(Path(sys.argv[1]).read_text());started=perf_counter()
    reply=dict(status='failed',error_code=None,metadata=dict(python=platform.python_version(),python_executable=sys.executable))
    try:
        if job['tool']!='uniface' or job['parameters']:raise WorkerFailure('invalid_input')
        reply['metadata'].update(check_gpu())
        import torch
        torch.cuda.reset_peak_memory_stats()
        reply['metadata'].update(generate(job,WorkerFailure));reply['status']='success'
    except WorkerFailure as e:reply['error_code']=str(e)
    except Exception:reply['error_code']='generation_failed';traceback.print_exc()
    reply['metadata']['elapsed_ms']=(perf_counter()-started)*1000
    Path(job['reply']).write_text(json.dumps(reply,indent=2,allow_nan=False)+'\n')
    return 0 if reply['status']=='success' else 1


if __name__=='__main__':raise SystemExit(main())
