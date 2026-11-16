"""Ports owned by shared; implementations are injected by core."""
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from shared.schemas import (Artifact, AttackRequest, Candidate, DetectorResult, ExaminerResult,
                            EmbeddingSpec, EvaluationResult, HealthResult, InputAsset, InputPolicy, ModifierOperation, Plan, ReferenceQuery, ReferenceRecord, ReferenceResult, ToolManifest)


class ModuleFailure(Exception):
    def __init__(self, code: str, message: str, *, field: str | None = None):
        self.code = code
        self.field = field
        super().__init__(message)


class InputIngestor(Protocol):
    def ingest(self, inputs: list[InputAsset], output_dir: Path) -> list[Artifact]: ...


class ArtifactAccess(Protocol):
    """Server-owned, confined IO injected into lightweight in-process tools."""
    policy: InputPolicy
    def read(self, artifact: Artifact) -> bytes: ...
    def write(self, data: bytes, output_dir: Path, *, filename: str) -> Path: ...


class ToolRegistry(Protocol):
    def manifests(self) -> list[ToolManifest]: ...
    def is_available(self, name: str) -> bool: ...


class ModifierCatalog(Protocol):
    """Metadata-only planning port; prepare validates without reading image pixels."""
    def manifests(self) -> list[ToolManifest]: ...
    def prepare(self, *, chain=None, preset=None, input_type: str = "image") -> tuple[list[ModifierOperation], dict[str, JsonValue]]: ...


class Planner(Protocol):
    def plan(self, request: AttackRequest, tool_registry: ToolRegistry) -> Plan: ...


class LLMProvider(Protocol):
    """Sanitized text-only JSON advice; has no tool or Gate authority."""
    def complete_json(self, task, payload, output_schema): ...


class GeneratorTool(Protocol):
    manifest: ToolManifest
    # Internal port: may raise ModuleFailure. Dispatch through Registry.invoke -> ModuleResult.
    # Implementations read validated snapshots, never request.inputs[*].path.
    def generate(self, request: AttackRequest, plan: Plan, output_dir: Path, *,
                 input_artifacts: list[Artifact]) -> Candidate: ...


class Modifier(Protocol):
    manifest: ToolManifest
    def modify(self, candidate: Candidate, output_dir: Path, *, parameters: dict[str, JsonValue] | None = None) -> Candidate: ...


class MetadataRetriever(Protocol):
    def search(self, query: ReferenceQuery) -> ReferenceResult: ...


class ReferenceBank(Protocol):
    def retrieve(self, query: ReferenceQuery) -> list[ReferenceRecord]: ...


@dataclass(frozen=True)
class ImageEmbedding:
    """Private internal vector; excluded from ReferenceResult/public serialization."""
    source_sha256: str
    vector: tuple[float, ...]
    elapsed_ms: float


class EmbeddingEncoder(Protocol):
    spec: EmbeddingSpec
    def encode(self, images: list[tuple[Artifact, bytes]]) -> list[ImageEmbedding]:
        """Consume only authorized, gateway-validated bytes, never reopen source paths."""
        ...


class Examiner(Protocol):
    def examine(self, request: AttackRequest, candidate: Candidate,
                references: list[ReferenceRecord]) -> ExaminerResult: ...


@dataclass(frozen=True)
class FaceObservation:
    source_sha256: str
    face_count: int
    elapsed_ms: float
    method: str
    runtime: dict[str, str]
    mock: bool = False
    details: dict[str, JsonValue] | None = None


class FacePresenceDetector(Protocol):
    def detect(self, artifact: Artifact, data: bytes) -> FaceObservation:
        """Consume authorized validated bytes, never reopen the original input path."""
        ...


class Refiner(Protocol):
    def refine(self, candidate: Candidate, examination: ExaminerResult,
               references: list[ReferenceRecord], output_dir: Path) -> Candidate: ...


class DetectorClient(Protocol):
    def health_check(self) -> HealthResult: ...
    def infer_one(self, image_artifact: Artifact) -> DetectorResult: ...
    def infer_batch(self, image_artifacts: list[Artifact]) -> list[DetectorResult]: ...


class Evaluator(Protocol):
    def evaluate(self, request: AttackRequest, candidate: Candidate) -> EvaluationResult: ...
