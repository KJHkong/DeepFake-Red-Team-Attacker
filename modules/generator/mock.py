import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from shared.contracts import ModuleFailure
from shared.output_paths import dated_name
from shared.schemas import Artifact, AttackRequest, Candidate, Plan, Taxonomy, ToolManifest


def image_artifact(path: Path, stage: str, parents=()) -> Artifact:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
    return Artifact(path=str(path.resolve()), media_type="image/png", stage=stage,
                    width=width, height=height, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    mock=True, visibility="mock_demo", display_allowed=True, parent_ids=list(parents))


class MockGenerator:
    manifest = ToolManifest(
        name="mock_image_generator", version="0.1", taxonomy=list(Taxonomy),
        input_schema=AttackRequest.model_json_schema(), output_schema=Candidate.model_json_schema(),
        backend="pillow_geometric_fixture", mock=True, mock_supported=True, ready_for_registry=True,
        estimated_duration_s=0,
        controllable_parameters=["seed"], capabilities=["workflow_fixture_only", "image"],
    )

    def generate(self, request: AttackRequest, plan: Plan, output_dir: Path, *,
                 input_artifacts: list[Artifact] | None = None) -> Candidate:
        if request.inputs or input_artifacts:
            raise ModuleFailure("input_role_mismatch", "Mock geometric generator does not consume image inputs", field="inputs")
        if request.mock_scenario == "generation_failed":
            raise ModuleFailure("generation_failed", "Injected mock generator failure")
        output_dir.mkdir(parents=True, exist_ok=True)
        # Stylized geometric test face; never claim this is a real Generator output.
        image = Image.new("RGB", (256, 256), (35, 48, 70))
        draw = ImageDraw.Draw(image)
        tint = request.seed % 60
        draw.ellipse((58, 30, 198, 216), fill=(185+tint, 148, 115))
        draw.ellipse((89, 92, 105, 108), fill=(22, 30, 40))
        draw.ellipse((151, 92, 167, 108), fill=(22, 30, 40))
        draw.arc((98, 125, 158, 174), 0, 180, fill=(85, 40, 40), width=4)
        draw.text((12, 230), "MOCK fixture / not a real fake", fill="white")
        path = output_dir / f"{dated_name('generator', 'mock_generated')}.png"
        image.save(path)
        return Candidate(artifact=image_artifact(path, "generator"), mock=True)
