"""Reuse official REFace wrapper; output remains its actual aligned512 face, no paste-back."""
import importlib.util
import json
from pathlib import Path
import sys


def generate(job,failure):
    repo,cache,out=Path(job['repo']),Path(job['checkpoint']),Path(job['output']).parent
    sys.path.insert(0,str(repo))
    spec=importlib.util.spec_from_file_location('day6_reface_reproduction',repo/'reproduction/run_reproduction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.O=out/'v2_day6_pipeline';module.C=cache;(module.O/'generated').mkdir(parents=True)
    configuration=dict(experiment_id='v2_day6_private_1026',source=job['inputs']['source'],target=job['inputs']['target'],**job['parameters'])
    try:run,work=module.prep(configuration)
    except (IndexError,TypeError) as exc:raise failure('no_face') from exc
    inputs=json.loads((run/'inputs.json').read_text())
    for row in inputs:row['provenance']='validated Run input snapshot from fixed real eval50; internal research'
    module.write(run/'inputs.json',inputs)
    model,config,load_report=module.build_model(work);module.write(run/'v2_day6_checkpoint_load_1026.json',load_report)
    result=module.run_one(configuration,run,work,model,config)
    Path(job['output']).write_bytes((run/'generated.png').read_bytes())
    return dict(native_face_size=[512,512],paste_back=False,output_scope='aligned face crop, not original Target canvas',
        source_reference=str(run/'inputs.json'),sampling_seconds=result['sampler_seconds'],seed=configuration['seed'],
        steps=configuration['steps'],guidance_scale=configuration['guidance_scale'],
        checkpoint_load=str(run/'v2_day6_checkpoint_load_1026.json'),
        peak_gpu_allocated_bytes=int(result['gpu_peak_allocated_gib']*2**30),peak_gpu_reserved_bytes=int(result['gpu_peak_reserved_gib']*2**30))
