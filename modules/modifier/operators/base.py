from dataclasses import dataclass, field

from PIL import Image
from pydantic import JsonValue


@dataclass
class ImageTransform:
    """Private encoding description, not a second public result protocol."""
    image: Image.Image
    format: str = 'PNG'
    encode_options: dict[str, int | bool] = field(default_factory=dict)
    details: dict[str, JsonValue] = field(default_factory=dict)
