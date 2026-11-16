"""Bind official MobileFaceSwap intermediates to the existing Artifact graph."""
import hashlib
import json
from pathlib import Path

import numpy as np

from core.artifacts import LocalArtifactAccess
from shared.contracts import ModuleFailure
from shared.schemas import Artifact, InputPolicy


def attach_lineage(candidate, inputs, output_dir):
    folder = Path(output_dir).resolve()
    roles = {a.role: a for a in inputs}
    try:
        target = roles['target']
        if (candidate.artifact.width, candidate.artifact.height) != (target.width, target.height):
            raise ValueError('Native paste did not preserve the Target canvas')
        display = all(a.display_allowed for a in inputs)
        access = LocalArtifactAccess(InputPolicy(allowed_roots=[str(folder)]), output_root=folder)
        runtime = candidate.metadata['runtime']
        assets = {}
        for raw in runtime['intermediates']:
            path = Path(raw['path'])
            if path.parent != folder:
                raise ValueError('Unexpected intermediate directory')
            a = Artifact(**raw, stage='generator', mock=False, display_allowed=display)
            access.read(a)
            if path.name in assets:
                raise ValueError('Duplicate intermediate')
            assets[path.name] = a
        names = ['source_aligned', 'source_identity', 'target_aligned', 'aligned_output', 'learned_mask']
        if set(assets) != {f'mobilefaceswap_{n}_1109.png' for n in names}:
            raise ValueError('Missing native preprocessing evidence')
        transform = folder / 'mobilefaceswap_transform_1109.json'
        if runtime['transform_path'] != str(transform):
            raise ValueError('Unexpected transform reference')
        _, raw = access.reader.read_bytes(str(transform))
        if hashlib.sha256(raw).hexdigest() != runtime['transform_sha256']:
            raise ValueError('Transform changed')
        data = json.loads(raw)
        forward = np.asarray(data['target_forward'], dtype=float)
        inverse = np.asarray(data['target_inverse'], dtype=float)
        if forward.shape != (2,3) or inverse.shape != (2,3) or not np.isfinite([forward,inverse]).all():
            raise ValueError('Invalid alignment matrices')
        product = np.vstack([forward,[0,0,1]]) @ np.vstack([inverse,[0,0,1]])
        if not np.allclose(product, np.eye(3), atol=1e-4):
            raise ValueError('Non-invertible alignment')
        get = lambda name: assets[f'mobilefaceswap_{name}_1109.png']
        get('source_aligned').parent_ids = [roles['source'].artifact_id]
        get('source_identity').parent_ids = [get('source_aligned').artifact_id]
        get('target_aligned').parent_ids = [target.artifact_id]
        parents = [get('source_identity').artifact_id, get('target_aligned').artifact_id]
        get('aligned_output').parent_ids = parents
        get('learned_mask').parent_ids = parents
        candidate.artifact.parent_ids = [target.artifact_id, get('aligned_output').artifact_id, get('learned_mask').artifact_id]
        candidate.artifact.display_allowed = display
        candidate.metadata['native_intermediate_artifacts'] = [a.model_dump(mode='json') for a in assets.values()]
        return candidate
    except (ValueError, KeyError, TypeError, OSError, ModuleFailure):
        raise ModuleFailure('invalid_generator_output', 'MobileFaceSwap native canvas/lineage validation failed') from None


class MobileFaceSwapGenerator:
    def __init__(self, native):
        self.native = native
        self.settings = native.settings
        self.manifest = native.manifest

    def missing_assets(self):
        return self.native.missing_assets()

    def generate(self, request, plan, output_dir, *, input_artifacts):
        candidate = self.native.generate(request, plan, output_dir, input_artifacts=input_artifacts)
        return attach_lineage(candidate, input_artifacts, output_dir)
