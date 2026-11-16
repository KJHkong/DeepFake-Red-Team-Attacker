"""Run-local files and atomic state; no process-global run registry."""
import json
import hashlib
import errno
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from shared.output_paths import create_output_directory
from core.inputs import LocalImageIngestor
from shared.contracts import ModuleFailure
from shared.schemas import Artifact, InputPolicy, RunMetadata


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / ".run_index"
        self.index.mkdir(mode=0o700, exist_ok=True)

    def create(self, topic: str, content: str, *, timestamp: datetime | None = None) -> tuple[str, Path]:
        run_id = uuid4().hex
        directory = create_output_directory(self.root, topic, content, timestamp=timestamp)
        (self.index / run_id).symlink_to(Path("..") / directory.name, target_is_directory=True)
        return run_id, directory

    def _directory(self, run_id: str) -> Path:
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise FileNotFoundError(run_id)
        directory = (self.index / run_id).resolve(strict=True)
        if directory.parent != self.root or directory.name.startswith("."):
            raise FileNotFoundError(run_id)
        return directory

    def save(self, run: RunMetadata):
        directory = self._directory(run.run_id)
        temporary = directory / f".metadata-{uuid4().hex}.tmp"
        temporary.write_text(run.model_dump_json(indent=2) + "\n")
        os.replace(temporary, directory / "metadata.json")

    def get(self, run_id: str) -> RunMetadata:
        path = self._directory(run_id) / "metadata.json"
        return RunMetadata.model_validate_json(path.read_text())


class LocalArtifactAccess:
    """Reuse canonical safe image reads; publish private files atomically without replacement.

    POSIX dir_fd and link semantics: https://docs.python.org/3.11/library/os.html#os.link
    This does not choose which business files are authorized; core supplies a scoped policy.
    """
    def __init__(self, policy: InputPolicy, *, output_root: Path):
        self.policy = policy
        self.reader = LocalImageIngestor(policy)
        self.output_root = Path(output_root)
        if not self.output_root.is_absolute() or '..' in self.output_root.parts or self.output_root == Path('/'):
            raise ValueError('Artifact output root must be an absolute scoped server directory')

    def read(self, artifact: Artifact) -> bytes:
        if not artifact.sha256 or re.fullmatch(r'[0-9a-f]{64}', artifact.sha256) is None:
            raise ModuleFailure('invalid_artifact', 'A verified Artifact SHA-256 is required')
        _, data = self.reader.read_bytes(artifact.path)
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise ModuleFailure('input_changed', 'Artifact bytes no longer match the recorded SHA-256')
        if artifact.size_bytes is not None and len(data) != artifact.size_bytes:
            raise ModuleFailure('invalid_artifact', 'Artifact byte count does not match metadata')
        width, height, fmt = self.reader._decode(data)
        media = {'PNG':'image/png', 'JPEG':'image/jpeg', 'WEBP':'image/webp'}[fmt]
        if (artifact.media_type != media or (artifact.width is not None and artifact.width != width) or
                (artifact.height is not None and artifact.height != height)):
            raise ModuleFailure('invalid_artifact', 'Artifact format or dimensions do not match metadata')
        return data

    def _open_output_directory(self, path: Path) -> int:
        if (not path.is_absolute() or '..' in path.parts or not path.is_relative_to(self.output_root)):
            raise ModuleFailure('path_not_allowed', 'Output must be inside the authorized Run or Local Eval directory')
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        current = Path('/')
        try:
            for component in path.parts[1:]:
                current /= component
                try:
                    next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                except FileNotFoundError:
                    if not current.is_relative_to(self.output_root):
                        raise ModuleFailure('path_not_allowed', 'Output root parent does not exist') from None
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    def write(self, data: bytes, output_dir: Path, *, filename: str) -> Path:
        if re.fullmatch(r'[a-z0-9_]+\.(png|jpg)', filename) is None:
            raise ModuleFailure('invalid_path', 'Artifact filename is invalid')
        if not 0 < len(data) <= self.policy.max_file_bytes:
            raise ModuleFailure('file_too_large', 'Encoded output exceeds the configured byte limit')
        fd, temporary = None, '.artifact-' + uuid4().hex + '.tmp'
        try:
            fd = self._open_output_directory(Path(output_dir))
            image_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                               mode=0o600, dir_fd=fd)
            with os.fdopen(image_fd, 'wb') as destination:
                destination.write(data)
                destination.flush()
                os.fsync(destination.fileno())
            # link publishes a complete file atomically and refuses any existing name.
            os.link(temporary, filename, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            return Path(output_dir) / filename
        except FileExistsError:
            raise ModuleFailure('output_exists', 'Output file already exists; overwrite is forbidden') from None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ModuleFailure('path_not_allowed', 'Output path contains a symbolic link or invalid directory') from None
            raise ModuleFailure('output_write_failed', 'Artifact output could not be written') from None
        finally:
            if fd is not None:
                try:
                    os.unlink(temporary, dir_fd=fd)
                except FileNotFoundError:
                    pass
                os.close(fd)
