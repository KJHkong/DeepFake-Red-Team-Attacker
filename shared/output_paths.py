"""Readable output names. IDs and source-code filenames are independent."""
from datetime import datetime
from pathlib import Path
import re


def dated_name(topic: str, content: str, *, timestamp: datetime | None = None,
               sequence: int = 1) -> str:
    for part in (topic, content):
        if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", part) is None:
            raise ValueError("Output topics/content must use lowercase words separated by underscores")
    if sequence < 1:
        raise ValueError("Output sequence must be positive")
    date = (timestamp or datetime.now().astimezone()).astimezone().strftime("%m%d")
    number = f"_{sequence:02d}" if sequence > 1 else ""
    return f"{topic}_{content}{number}_{date}"


def create_output_directory(parent: Path, topic: str, content: str, *,
                            timestamp: datetime | None = None) -> Path:
    stamp = timestamp or datetime.now().astimezone()
    # Validate before touching the filesystem; mkdir resolves concurrent collisions.
    dated_name(topic, content, timestamp=stamp)
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        directory = parent / dated_name(topic, content, timestamp=stamp, sequence=sequence)
        try:
            directory.mkdir(mode=0o700)
            return directory
        except FileExistsError:
            sequence += 1
