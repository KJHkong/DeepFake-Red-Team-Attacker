"""Reuse the previously audited DiffSwap reproduction functions inside a private Run."""
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


def generate(job,failure):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    repo,cache,out=Path(job['repo']),Path(job['checkpoint']),Path(job['output']).parent
    sys.path.insert(0,str(repo));os.environ.update(TF_CPP_MIN_LOG_LEVEL='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',MPLBACKEND='Agg')
    spec=importlib.util.spec_from_file_location('day6_diffswap_reproduction',repo/'reproduction/run_reproduction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=out/'v2_day6_pipeline';module.CACHE=cache
    params=job['parameters'];torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    configuration=dict(experiment_id='v2_day6_private_1026',source=job['inputs']['source'],target=job['inputs']['target'],
        seed=params['seed'],steps=params['steps'],tgt_scale=params['tgt_scale'],crop_coordinate_fix=True,source_latent_fix=True)
    run,work=module.prepare(configuration)
    manifest=json.loads((run/'input_manifest.json').read_text())
    for row in manifest:row['provenance']='validated Run input snapshot from fixed real eval50; internal research'
    module.write_json(run/'input_manifest.json',manifest)
    started=time.perf_counter()
    try:pipeline=module.preprocess(work,configuration)
    except (KeyError,FileNotFoundError,RuntimeError) as exc:
        # Author preprocessor can reject/delete ONLY its own workspace RGB copies.
        if not all((work/'data/portrait/align'/role/(role+'.png')).is_file() for role in ('source','target')):
            raise failure('no_face') from exc
        raise
    prep_seconds=time.perf_counter()-started
    model,report=module.build_model();module.write_json(run/'v2_day6_checkpoint_load_1026.json',report)
    from torch.utils.data import DataLoader
    from ldm.data.portrait import Portrait
    from ldm.models.diffusion.ddim import DDIMSampler
    random.seed(params['seed']);np.random.seed(params['seed']);torch.manual_seed(params['seed']);torch.cuda.manual_seed_all(params['seed'])
    torch.cuda.reset_peak_memory_stats();sampling=time.perf_counter()
    batch=next(iter(DataLoader(Portrait(str(work/'data/portrait')),batch_size=1,num_workers=0)))
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    with torch.no_grad():
        original_forward=model.cond_stage_model.forward
        def source_condition(cond,swap=False):
            source=cond['image_src'].permute(0,3,1,2).contiguous().to(model.device)
            cond['z_src']=model.get_first_stage_encoding(model.encode_first_stage(source)).detach()
            return original_forward(cond,swap=swap)
        model.cond_stage_model.forward=source_condition
        z,c,_,_,_=model.get_input(batch,model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,swap=True)
        model.cond_stage_model.forward=original_forward
        sampler=DDIMSampler(model,tgt_scale=params['tgt_scale'])
        mask=(F.interpolate((1-batch['mask'].float())[:,None],size=z.shape[-2:],mode='nearest')>0).float()
        with model.ema_scope('Day6 fixed masked diffusion'):
            samples,_=sampler.sample(params['steps'],1,(model.channels,model.image_size,model.image_size),c,eta=0.,x0=z,mask=mask,verbose=False)
        decoded=model.decode_first_stage(samples)
        if not torch.isfinite(decoded).all():raise failure('generation_failed')
        pixels=(torch.clamp((decoded+1)/2,0,1).cpu().numpy()*255).astype(np.uint8).transpose(0,2,3,1)[0]
    raw=work/'data/portrait/swap_res'/f'diffswap_{sampler.tgt_scale}'/'source/target.png';raw.parent.mkdir(parents=True)
    Image.fromarray(pixels).save(raw);torch.cuda.synchronize();sample_seconds=time.perf_counter()-sampling
    pipeline.repair_by_mask();pipeline.paste()
    result=work/'data/portrait/swap_res_ori'/f'diffswap_{sampler.tgt_scale}'/'source/target.png'
    Path(job['output']).write_bytes(result.read_bytes())
    with Image.open(job['output']) as im:size=list(im.size)
    return dict(torch=torch.__version__,native_face_size=[256,256],paste_back=True,output_size=size,
        source_wiring_fix=True,crop_coordinate_fix=True,steps=200,tgt_scale=params['tgt_scale'],seed=params['seed'],
        preprocess_seconds=prep_seconds,sampling_seconds=sample_seconds,
        preprocessing='author dlib/FFHQ/MTCNN on Run-private RGB copies; author may resize canvas below400/above2000',
        workspace=str(work),checkpoint_load=str(run/'v2_day6_checkpoint_load_1026.json'),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
