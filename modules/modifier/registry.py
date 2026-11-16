"""Registered in-process tools, using existing manifests/results and confined IO."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from PIL import Image, __version__ as pillow_version
from pydantic import ValidationError

from modules.modifier import parameters as params
from modules.modifier.operators import (jpeg_recompress, gaussian_blur, color_adjust,
                                       gaussian_noise, resize_resample,sharpen,crop_pad,pixelate)
from shared.contracts import ArtifactAccess, ModuleFailure
from shared.output_paths import dated_name
from shared.schemas import Artifact, Candidate, ErrorInfo, ModuleResult, Status, ToolManifest, now

IMPLEMENTATIONS = {
    'jpeg_recompress': (params.JPEGParameters, jpeg_recompress.apply),
    'gaussian_blur': (params.BlurParameters, gaussian_blur.apply),
    'color_adjust': (params.ColorParameters, color_adjust.apply),
    'gaussian_noise': (params.NoiseParameters, gaussian_noise.apply),
    'resize_resample': (params.ResizeParameters, resize_resample.apply),
    'sharpen': (params.SharpenParameters, sharpen.apply),
    'crop_pad': (params.CropPadParameters, crop_pad.apply),
    'pixelate': (params.PixelateParameters, pixelate.apply),
}


def failure(source, stamp, elapsed, error, parameters=None, *, tool_name=None):
    return ModuleResult(module='modifier', status=Status(error.code) if error.code in Status._value2member_map_ else Status.MODULE_FAILED,
        mock=source.mock, started_at=stamp, elapsed_ms=elapsed, inputs=[source.artifact_id],
        parameters=parameters or {}, data={'operator':tool_name, 'output_generated':False},
        error=ErrorInfo(code=error.code, message=str(error), field=error.field))


class ImageModifierTool:
    def __init__(self, manifest, access):
        self.manifest, self.access = manifest, access
        self.parameter_type, self.transform = IMPLEMENTATIONS[manifest.name]

    def validate(self, parameters):
        try:
            return self.parameter_type.model_validate({**self.manifest.default_parameters, **(parameters or {})})
        except (ValidationError, TypeError):
            raise ModuleFailure('invalid_parameters', 'Modifier parameters violate the registered types or ranges',
                                field='modifier.parameters') from None

    def invoke(self, source: Artifact, output_dir: Path, *, parameters=None) -> ModuleResult:
        started, stamp = perf_counter(), now()
        resolved = {}
        try:
            if self.manifest.availability not in {'ready', 'available'}:
                raise ModuleFailure('unsupported_modifier', 'Modifier is not available')
            parsed = self.validate(parameters)
            resolved = parsed.model_dump()
            raw = self.access.read(source)
            with Image.open(BytesIO(raw)) as original:
                input_format, input_mode = original.format, original.mode
                before = original.convert('RGB')
            transformed = self.transform(before, parsed, self.access.policy.max_pixels)
            if transformed.image.width * transformed.image.height > self.access.policy.max_pixels:
                raise ModuleFailure('image_too_large', 'Modifier output exceeds the configured pixel limit')
            encoded = BytesIO()
            transformed.image.save(encoded, format=transformed.format, **transformed.encode_options)
            data = encoded.getvalue()
            if len(data) > self.access.policy.max_file_bytes:
                raise ModuleFailure('file_too_large', 'Encoded output exceeds the configured byte limit')
            with Image.open(BytesIO(data)) as check:
                check.verify()
            with Image.open(BytesIO(data)) as check:
                check.load()
                actual_format = check.format
                after = check.convert('RGB')
            geometry_changed = before.size != after.size
            changed, mae = None, None
            if not geometry_changed:
                difference = np.abs(np.asarray(before, dtype=np.int16) - np.asarray(after, dtype=np.int16))
                changed, mae = bool(np.any(difference)), float(np.mean(difference))
            suffix, media = {'PNG':('.png', 'image/png'), 'JPEG':('.jpg', 'image/jpeg')}[actual_format]
            for sequence in range(1, 1001):
                try:
                    path = self.access.write(data, output_dir,
                        filename=dated_name('modifier', self.manifest.name, sequence=sequence) + suffix)
                    break
                except ModuleFailure as exc:
                    if exc.code != 'output_exists':
                        raise
            else:
                raise ModuleFailure('output_write_failed', 'Output naming collision limit exceeded')
            artifact = Artifact(path=str(path), media_type=media, stage='modifier', mock=source.mock,
                width=after.width, height=after.height, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                parent_ids=[source.artifact_id], visibility=source.visibility, display_allowed=source.display_allowed)
            metadata = {'operator':self.manifest.name, 'tool_version':self.manifest.version,
                'parameters':resolved, 'input_format':input_format, 'input_mode':input_mode, 'output_format':actual_format,
                'input_sha256':source.sha256, 'output_sha256':artifact.sha256,
                'normalization':'RGB conversion; no EXIF transpose; no input metadata copied',
                'input_dimensions':list(before.size), 'output_dimensions':list(after.size),
                'geometry_changed':geometry_changed, 'pixel_changed':changed, 'pixel_mae':mae,
                'pixel_comparison':'not_comparable_dimensions' if geometry_changed else 'decoded RGB, absolute channel difference',
                'bytes_changed':raw != data, 'effect_status':'geometry_changed' if geometry_changed else 'pixels_changed' if changed else 'no_pixel_change',
                'operator_details':transformed.details, 'execution':'in_process_python',
                'runtime':{'python':platform.python_version(), 'pillow':pillow_version, 'numpy':np.__version__},
                'quality_check':'not_evaluated', 'detector_executed':False}
            candidate = Candidate(artifact=artifact, mock=source.mock, metadata=metadata)
            return ModuleResult(module='modifier', status=Status.SUCCESS, mock=source.mock,
                started_at=stamp, elapsed_ms=(perf_counter()-started)*1000, parameters=resolved,
                inputs=[source.artifact_id], outputs=[artifact.artifact_id], data=metadata, candidate=candidate)
        except ModuleFailure as exc:
            return failure(source, stamp, (perf_counter()-started)*1000, exc, resolved, tool_name=self.manifest.name)
        except OSError:
            return failure(source, stamp, (perf_counter()-started)*1000,
                           ModuleFailure('output_write_failed', 'Modifier output could not be encoded or written'), resolved, tool_name=self.manifest.name)
        except Exception:
            return failure(source, stamp, (perf_counter()-started)*1000,
                           ModuleFailure('modifier_failed', 'Modifier execution failed'), resolved, tool_name=self.manifest.name)

    def modify(self, candidate: Candidate, output_dir: Path, *, parameters=None) -> Candidate:
        """Compatibility with the existing Modifier port; typed callers use invoke."""
        result = self.invoke(candidate.artifact, output_dir, parameters=parameters)
        if result.error:
            raise ModuleFailure(result.error.code, result.error.message, field=result.error.field)
        return result.candidate.model_copy(update={'revision':candidate.revision,
            'metadata':{**candidate.metadata, **result.candidate.metadata, 'upstream_candidate_id':candidate.candidate_id}})


class ModifierRegistry:
    def __init__(self, access: ArtifactAccess):
        self.access = access
        self._tools = {}
        metadata = Path(__file__).parent / 'metadata/real_tools.json'
        for entry in json.loads(metadata.read_text()):
            manifest = ToolManifest.model_validate(entry)
            self.register(ImageModifierTool(manifest, access))

    def register(self, tool):
        if tool.manifest.kind != 'modifier' or tool.manifest.mock or tool.manifest.name in self._tools:
            raise ValueError('A unique real Modifier manifest is required')
        self._tools[tool.manifest.name] = tool

    def get(self, name):
        if name not in self._tools or not self.is_available(name):
            raise ModuleFailure('unsupported_modifier', 'Modifier is not registered or available', field='modifier.tool_name')
        return self._tools[name]

    def is_available(self, name):
        return name in self._tools and self._tools[name].manifest.availability in {'ready','available'}

    def manifest(self, name):
        return self.get(name).manifest

    def manifests(self):
        return [tool.manifest for tool in self._tools.values()]

    def validate(self, name, parameters=None):
        return self.get(name).validate(parameters).model_dump()

    def invoke(self, name, source: Artifact, output_dir: Path, *, parameters=None):
        try:
            tool = self.get(name)
        except ModuleFailure as exc:
            return failure(source, now(), 0, exc, tool_name=name)
        return tool.invoke(source, output_dir, parameters=parameters)
