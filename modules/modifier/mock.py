import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from shared.output_paths import dated_name
from shared.contracts import ModuleFailure
from shared.schemas import Artifact, Candidate, Taxonomy, ToolManifest


class MockModifier:
    manifest = ToolManifest(name="mock_image_modifier",version="0.1",kind="modifier",taxonomy=list(Taxonomy),
        input_schema=Candidate.model_json_schema(),output_schema=Candidate.model_json_schema(),
        backend="pillow_geometric_fixture",mock=True,mock_supported=True,ready_for_registry=True,
        capabilities=["mock_border"],estimated_duration_s=0)

    def modify(self, candidate: Candidate, output_dir: Path, *, parameters: dict | None = None) -> Candidate:
        if parameters:
            raise ModuleFailure("unsupported_parameter", "Mock modifier accepts no parameters", field="modifier.parameters")
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(candidate.artifact.path) as source:
            image = source.convert("RGB")
        ImageDraw.Draw(image).rectangle((6, 6, 249, 249), outline=(80, 180, 210), width=3)
        path = output_dir / f"{dated_name('modifier', 'mock_modified')}.png"
        image.save(path)
        artifact = Artifact(path=str(path.resolve()), media_type="image/png", stage="modifier",
                            mock=True, width=image.width, height=image.height,
                            parent_ids=[candidate.artifact.artifact_id], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            display_allowed=True, visibility="mock_demo")
        return Candidate(artifact=artifact, revision=candidate.revision, mock=True)
