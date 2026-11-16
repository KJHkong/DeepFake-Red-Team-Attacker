"""Allowlisted local image ingestion. No scanning, URLs, external calls or source edits.

Linux descriptor-relative opens reject symlinks at every path component, including
ancestors. Reference: https://docs.python.org/3.11/library/os.html#os.open
Image.verify() plus reopened load():
https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.verify
"""
import errno
import hashlib
from io import BytesIO
import os
from pathlib import Path
import stat
import warnings

from PIL import Image

from shared.contracts import ModuleFailure
from shared.output_paths import dated_name
from shared.schemas import Artifact, InputAsset, InputPolicy


def load_input_policy(path: Path) -> InputPolicy:
    policy = InputPolicy.model_validate_json(path.read_text())
    # Only the trusted server configuration may resolve aliases for allowed roots.
    return policy.model_copy(update={"allowed_roots": [
        str((path.parent / root).resolve()) for root in policy.allowed_roots]})


class LocalImageIngestor:
    def __init__(self, policy: InputPolicy):
        self.policy = policy
        self.roots = [Path(p) for p in policy.allowed_roots]
        if any(not p.is_absolute() or ".." in p.parts for p in self.roots):
            raise ValueError("Input policy roots must be absolute server-controlled directories")

    def read_bytes(self, value: str) -> tuple[Path, bytes]:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ModuleFailure("invalid_path", "Image path must contain valid Unicode") from None
        if "\0" in value:
            raise ModuleFailure("invalid_path", "Image path contains an invalid character")
        path = Path(value)
        if not path.is_absolute():
            raise ModuleFailure("invalid_path", "Image paths must be absolute local paths")
        if ".." in path.parts or not any(path.is_relative_to(root) for root in self.roots):
            raise ModuleFailure("path_not_allowed", "Image path is outside the allowed input directories or contains traversal")
        if path == Path("/"):
            raise ModuleFailure("invalid_file", "Input must be a regular image file")
        # Do not resolve an untrusted path and reopen it: that permits symlink races.
        directory_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for component in path.parts[1:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                  dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            image_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                               dir_fd=directory_fd)
            with os.fdopen(image_fd, "rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ModuleFailure("invalid_file", "Input must be a regular image file")
                if before.st_nlink != 1:
                    raise ModuleFailure("path_not_allowed", "Hard-linked input aliases are not permitted; submit a standalone file")
                if before.st_size > self.policy.max_file_bytes:
                    raise ModuleFailure("file_too_large", "Image exceeds the configured file byte limit")
                data = source.read(self.policy.max_file_bytes + 1)
                after = os.fstat(source.fileno())
                if len(data) > self.policy.max_file_bytes:
                    raise ModuleFailure("file_too_large", "Image exceeds the configured file byte limit")
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ModuleFailure("input_changed", "Input changed while being read; submit a stable file")
                return path, data
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise ModuleFailure("file_not_found", "Input image does not exist") from None
            if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EACCES, errno.EPERM}:
                raise ModuleFailure("path_not_allowed", "Input path is inaccessible or contains a disallowed symbolic link") from None
            raise ModuleFailure("invalid_file", "Input file cannot be read") from None
        finally:
            os.close(directory_fd)

    def _decode(self, data: bytes) -> tuple[int, int, str]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data), formats=self.policy.allowed_formats) as image:
                    width, height = image.size
                    if width * height > self.policy.max_pixels:
                        raise ModuleFailure("image_too_large", "Image exceeds the configured pixel limit")
                    if getattr(image, "n_frames", 1) != 1:
                        raise ModuleFailure("invalid_image", "Only single-frame images are supported")
                    image_format = image.format
                    image.verify()
                with Image.open(BytesIO(data), formats=self.policy.allowed_formats) as image:
                    image.load()
            return width, height, image_format
        except ModuleFailure:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ModuleFailure("image_too_large", "Image exceeds safe decoder limits") from None
        except Exception:
            # Decoder/plugin failures are untrusted input errors; never expose paths.
            raise ModuleFailure("invalid_image", "Image cannot be fully decoded as a supported PNG, JPEG or WEBP") from None

    def ingest(self, inputs: list[InputAsset], output_dir: Path) -> list[Artifact]:
        validated, total = [], 0
        for index, item in enumerate(inputs):
            try:
                path, data = self.read_bytes(item.path)
                total += len(data)
                if total > self.policy.max_total_bytes:
                    raise ModuleFailure("inputs_too_large", "Inputs exceed the configured total byte limit")
                width, height, image_format = self._decode(data)
                validated.append((item, path, data, width, height, image_format))
            except ModuleFailure as exc:
                raise ModuleFailure(exc.code, str(exc), field=f"inputs.{index}.path") from None
        if not validated:
            return []
        # Snapshot only explicitly submitted, validated files, after all inputs pass.
        output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        artifacts = []
        for index, (item, source, data, width, height, image_format) in enumerate(validated, 1):
            suffix, media = {"PNG": (".png", "image/png"), "JPEG": (".jpg", "image/jpeg"), "WEBP": (".webp", "image/webp")}[image_format]
            path = output_dir / (dated_name("input", f"{item.role}_{index:02d}") + suffix)
            with path.open("xb") as destination:
                destination.write(data)
            path.chmod(0o600)
            artifacts.append(Artifact(path=str(path.resolve()), media_type=media, stage="input", mock=False,
                role=item.role, source_path=str(source), width=width, height=height, size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(), display_allowed=item.display_allowed,
                visibility="internal", provenance="local_input_snapshot"))
        return artifacts
