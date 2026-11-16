"""Server-owned catalog. Loading metadata never imports GPU/ONNX dependencies."""
import json
import os
from pathlib import Path

from modules.generator.adapters.isolated import IsolatedGenerator, WorkerSettings
from modules.generator.adapters.parameters import PARAMETERS
from modules.generator.mock import MockGenerator
from modules.generator.registry import GeneratorRegistry, pending_manifests
from modules.modifier.mock import MockModifier
from shared.schemas import ToolManifest

ROOT = Path(__file__).resolve().parents[2]


def load_catalog() -> list[ToolManifest]:
    path = Path(os.environ.get("ATTACKER_TOOL_CATALOG") or ROOT/"modules/generator/metadata/real_tools.json")
    return [ToolManifest.model_validate(m) for m in json.loads(path.read_text())]


def build_generator_registry(*, modifier=None, worker_config: Path | None = None) -> GeneratorRegistry:
    """Only server configuration chooses executable, checkpoint, device and deadlines."""
    registry = GeneratorRegistry()
    registry.register(MockGenerator())
    registry.register_modifier(modifier or MockModifier())
    for manifest in pending_manifests():
        registry.declare(manifest)
    profiles = json.loads(Path(worker_config or os.environ.get("ATTACKER_WORKER_CONFIG") or ROOT/"configs/generator_workers.json").read_text())
    for manifest in load_catalog():
        if manifest.name not in profiles or manifest.availability in {"manifest_only", "disabled"}:
            if manifest.availability in {"ready", "available"}:
                manifest.availability, manifest.ready_for_registry = "not_ready", False
                manifest.planner_notes.append("No worker profile configured on this host.")
            registry.declare(manifest)
            continue
        settings = WorkerSettings(worker=str(ROOT/"modules/generator/workers/run.py"), **profiles[manifest.name])
        manifest.parameter_schema = PARAMETERS[manifest.name].model_json_schema()
        manifest.worker_hint = f"{settings.python} -B {settings.worker} <private-job.json>"
        tool = IsolatedGenerator(manifest, settings)
        if tool.missing_assets():
            manifest.availability, manifest.ready_for_registry = "not_ready", False
            manifest.planner_notes.append("Required local assets missing; execution is unavailable on this host.")
            registry.declare(manifest)
        else:
            manifest.availability, manifest.ready_for_registry = "available", True
            registry.register(tool)
    return registry
