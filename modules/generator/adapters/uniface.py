"""UniFace crop/full-canvas lineage; existing FFHQ inverse geometry is reused."""
import hashlib
import json
from pathlib import Path

import numpy as np

from core.artifacts import LocalArtifactAccess
from modules.generator.adapters.v2_day6_reface_pasteback import alignment_matrices
from shared.contracts import ModuleFailure
from shared.schemas import Artifact, InputPolicy


def attach_lineage(candidate, inputs, output_dir):
    folder=Path(output_dir).resolve()
    access=LocalArtifactAccess(InputPolicy(allowed_roots=[str(folder)]),output_root=folder)
    try:
        roles={a.role:a for a in inputs};target=roles['target'];display=all(a.display_allowed for a in inputs)
        if (candidate.artifact.width,candidate.artifact.height)!=(target.width,target.height):
            raise ValueError('UniFace full-canvas output dimensions mismatch')
        runtime=candidate.metadata['runtime'];assets={}
        for raw in runtime['intermediates']:
            p=Path(raw['path'])
            if p.parent!=folder or p.name in assets:raise ValueError('Unexpected intermediate path')
            a=Artifact(**raw,stage='generator',mock=False,display_allowed=display)
            access.read(a);assets[p.name]=a
        names=['source_aligned','target_aligned','aligned_output','target_parser','projected_alpha']
        if set(assets)!={f'uniface_{n}_1109.png' for n in names}:raise ValueError('Missing UniFace evidence')
        path=folder/'uniface_transform_1109.json'
        if runtime['transform_path']!=str(path):raise ValueError('Unexpected transform path')
        _,raw=access.reader.read_bytes(str(path))
        if hashlib.sha256(raw).hexdigest()!=runtime['transform_sha256']:raise ValueError('Geometry changed')
        data=json.loads(raw);forward,inverse=alignment_matrices(data['quad_original'],data['alignment_size'])
        if not np.allclose(forward,data['target_to_aligned'],atol=1e-6) or not np.allclose(inverse,data['aligned_to_target'],atol=1e-6):
            raise ValueError('Inverse alignment mismatch')
        if data['outside_mask_changed_pixels']!=0:raise ValueError('Background changed outside blend mask')
        get=lambda name:assets[f'uniface_{name}_1109.png']
        get('source_aligned').parent_ids=[roles['source'].artifact_id]
        get('target_aligned').parent_ids=[target.artifact_id]
        get('aligned_output').parent_ids=[get('target_aligned').artifact_id,get('source_aligned').artifact_id]
        get('target_parser').parent_ids=[get('target_aligned').artifact_id]
        get('projected_alpha').parent_ids=[target.artifact_id,get('target_parser').artifact_id]
        candidate.artifact.parent_ids=[target.artifact_id,get('aligned_output').artifact_id,get('projected_alpha').artifact_id]
        candidate.artifact.display_allowed=display
        candidate.metadata['native_intermediate_artifacts']=[a.model_dump(mode='json') for a in assets.values()]
        return candidate
    except (ValueError,KeyError,TypeError,OSError,ModuleFailure):
        raise ModuleFailure('invalid_generator_output','UniFace native geometry/lineage validation failed') from None


class UniFaceGenerator:
    def __init__(self,native):
        self.native=native;self.settings=native.settings;self.manifest=native.manifest
    def missing_assets(self):return self.native.missing_assets()
    def generate(self,request,plan,output_dir,*,input_artifacts):
        candidate=self.native.generate(request,plan,output_dir,input_artifacts=input_artifacts)
        return attach_lineage(candidate,input_artifacts,output_dir)
