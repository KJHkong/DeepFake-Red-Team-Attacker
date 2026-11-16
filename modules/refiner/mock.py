import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from shared.contracts import ModuleFailure
from shared.output_paths import dated_name
from shared.schemas import Artifact, Candidate, ExaminerResult, ReferenceRecord


class MockRefiner:
    def refine(self, candidate: Candidate, examination: ExaminerResult,
               references: list[ReferenceRecord], output_dir: Path) -> Candidate:
        if candidate.revision != 0:
            raise ModuleFailure("invalid_input", "Only one refinement permitted")
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(candidate.artifact.path) as source:
            image = source.convert("RGB")
        ImageDraw.Draw(image).rectangle((6, 6, 249, 249), outline=(90, 210, 130), width=4)
        path = output_dir / f"{dated_name('refiner', 'mock_refined')}.png"
        image.save(path)
        artifact = Artifact(path=str(path.resolve()), media_type="image/png", stage="refiner", mock=True,
                            width=image.width, height=image.height, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            parent_ids=[candidate.artifact.artifact_id], visibility="mock_demo", display_allowed=True)
        return Candidate(artifact=artifact, revision=1, mock=True)
