"""Fixed argv + JSON IPC; model dependencies never enter the API interpreter.

Process isolation/timeout: https://docs.python.org/3.11/library/subprocess.html#subprocess.Popen
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from time import perf_counter

from PIL import Image
from pydantic import Field, JsonValue, ValidationError

from modules.generator.adapters.parameters import PARAMETERS
from shared.contracts import ModuleFailure
from shared.output_paths import dated_name
from shared.schemas import Artifact, Candidate, Contract


@dataclass(frozen=True)
class WorkerSettings:
    python: str
    worker: str
    repo: str
    checkpoint: str
    timeout_s: float = 60
    gpu: str = "unconfigured"
    required_files: tuple[str, ...] = ()


class WorkerReply(Contract):
    status: str
    error_code: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class IsolatedGenerator:
    def __init__(self, manifest, settings: WorkerSettings):
        self.manifest, self.settings = manifest, settings

    def missing_assets(self) -> list[str]:
        paths = [self.settings.python, self.settings.worker, self.settings.repo,
                 self.settings.checkpoint, *self.settings.required_files]
        return [p for p in paths if not Path(p).exists()]

    def _validated_parameters(self, request, plan):
        parameters = PARAMETERS[self.manifest.name].model_validate(plan.parameters).model_dump()
        if self.manifest.name=='liveportrait_static':
            from modules.generator.adapters.static_actions import controls_for
            controls_for(parameters['action_id'],request.prompt)
        return parameters

    def generate(self, request, plan, output_dir, *, input_artifacts):
        try:
            parameters = self._validated_parameters(request, plan)
        except (ValidationError, KeyError, ValueError):
            raise ModuleFailure("invalid_parameters", "Generator parameter values are invalid", field="parameters") from None
        if self.missing_assets():
            raise ModuleFailure("tool_unavailable", "Required local worker assets are missing", field="generator")
        # The only image paths forwarded are immutable snapshots produced by InputIngestor.
        if len(input_artifacts) != len(request.inputs) or any(
                artifact.provenance != "local_input_snapshot" or artifact.mock or artifact.role != asset.role
                for artifact, asset in zip(input_artifacts, request.inputs)):
            raise ModuleFailure("invalid_input", "Validated image snapshots are required", field="inputs")
        for artifact in input_artifacts:
            path = Path(artifact.path)
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 20*1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                raise ModuleFailure("input_changed", "Validated input snapshot is missing or changed", field="inputs")
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        output = output_dir / (dated_name("generator", self.manifest.name+"_output")+".png")
        job_path, reply_path = output_dir / "job.json", output_dir / "worker_result.json"
        job = {"tool": self.manifest.name, "repo": self.settings.repo, "checkpoint": self.settings.checkpoint,
            "parameters": parameters, "prompt": request.prompt,
            "inputs": {a.role: a.path for a in input_artifacts}, "output": str(output), "reply": str(reply_path)}
        job_path.write_text(json.dumps(job, indent=2)+"\n")
        os.chmod(job_path, 0o600)
        # Explicit offline flags and no inherited credentials; all models are server configured.
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TZ") if k in os.environ}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
            WANDB_DISABLED="true", DO_NOT_TRACK="1", PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="" if self.manifest.name == "inswapper" else self.settings.gpu,
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            TORCH_EXTENSIONS_DIR=str(output_dir/"torch_extensions"))
        start = perf_counter()
        with (output_dir / "worker.log").open("x") as log:
            process = subprocess.Popen([self.settings.python, "-B", self.settings.worker, str(job_path)],
                cwd=output_dir, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=min(self.settings.timeout_s, request.budget.max_duration_s))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Worker exited while its deadline was being enforced.
                process.wait()
                raise ModuleFailure("generator_timeout", "Generator worker timed out; no valid output is claimed") from None
        if not reply_path.is_file() or reply_path.is_symlink() or reply_path.stat().st_size > 1024**2:
            raise ModuleFailure("invalid_generator_output", "Worker did not return a valid result envelope")
        try:
            reply = WorkerReply.model_validate_json(reply_path.read_text())
        except (ValidationError, OSError):
            raise ModuleFailure("invalid_generator_output", "Worker result envelope is invalid") from None
        if code or reply.status != "success":
            known = {"no_face", "safety_filtered", "gpu_busy", "generation_failed", "invalid_input"}
            failure = reply.error_code if reply.error_code in known else "generation_failed"
            raise ModuleFailure(failure, {"no_face": "No usable face in source or target", "safety_filtered": "Model safety checker filtered the output",
                "gpu_busy": "Configured GPU is busy or has insufficient free memory"}.get(failure, "Generator worker failed; inspect private metadata"))
        try:
            if output.is_symlink() or not output.is_file() or not 0 < output.stat().st_size <= 20*1024**2:
                raise ValueError("Invalid output file")
            with Image.open(output) as image:
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1 or image.width*image.height > 25_000_000:
                    raise ValueError("Invalid output format")
                image.verify()
            with Image.open(output) as image:
                image.load()
                width, height = image.size
        except (OSError, ValueError, Image.DecompressionBombError):
            raise ModuleFailure("invalid_generator_output", "Generator output is not a valid bounded PNG image") from None
        artifact = Artifact(path=str(output), media_type="image/png", stage="generator", mock=False,
            width=width, height=height, size_bytes=output.stat().st_size,
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(), parent_ids=[a.artifact_id for a in input_artifacts],
            display_allowed=all(a.display_allowed for a in input_artifacts))
        return Candidate(artifact=artifact, mock=False, metadata={"worker_result_path": str(reply_path),
            "worker_log_path": str(output_dir/"worker.log"), "worker_elapsed_ms": (perf_counter()-start)*1000,
            "tool_name": self.manifest.name, "tool_version": self.manifest.version, "parameters": parameters,
            "runtime": reply.metadata, "generator_executed": True, "detector_executed": False})
