"""Bind native preprocessing/compositing images to the existing Artifact graph."""
from pathlib import Path
from core.artifacts import LocalArtifactAccess
from shared.schemas import Artifact,InputPolicy


class NativeLineageGenerator:
    def __init__(self,native):
        self.native=native;self.settings=native.settings;self.manifest=native.manifest
    def missing_assets(self):return self.native.missing_assets()
    def generate(self,request,plan,output_dir,*,input_artifacts):
        candidate=self.native.generate(request,plan,output_dir,input_artifacts=input_artifacts)
        if self.manifest.name!='blendface':return candidate
        access=LocalArtifactAccess(InputPolicy(allowed_roots=[str(Path(output_dir).resolve())]),output_root=Path(output_dir).resolve())
        by_name={}
        for raw in candidate.metadata['runtime']['intermediates']:
            asset=Artifact(**raw,stage='generator',mock=False,display_allowed=candidate.artifact.display_allowed)
            access.read(asset);by_name[Path(asset.path).name]=asset
        get=lambda name:by_name[f'generator_blendface_{name}_1026.png']
        roles={a.role:a for a in input_artifacts}
        get('source_aligned').parent_ids=[roles['source'].artifact_id]
        get('target_aligned').parent_ids=[roles['target'].artifact_id]
        get('raw_face').parent_ids=[get('source_aligned').artifact_id,get('target_aligned').artifact_id]
        get('learned_alpha').parent_ids=[get('target_aligned').artifact_id]
        get('aligned_output').parent_ids=[get('raw_face').artifact_id,get('learned_alpha').artifact_id,get('target_aligned').artifact_id]
        candidate.artifact.parent_ids=[roles['target'].artifact_id,get('raw_face').artifact_id,get('learned_alpha').artifact_id]
        candidate.metadata['native_intermediate_artifacts']=[a.model_dump(mode='json') for a in by_name.values()]
        return candidate
