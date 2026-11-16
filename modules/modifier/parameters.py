"""Strict v0.1 ranges, independent of ImageNet-C severity."""
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator

from shared.schemas import Contract


class Parameters(Contract):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class JPEGParameters(Parameters):
    quality: int = Field(default=75, ge=20, le=95)
    subsampling: Literal[0, 1, 2] = 2

    @field_validator('subsampling', mode='before')
    @classmethod
    def sampling_is_integer(cls, value):
        if type(value) is not int:
            raise ValueError('subsampling must be an integer enum')
        return value


class BlurParameters(Parameters):
    radius: float = Field(default=1.0, ge=.1, le=5.)


class ColorParameters(Parameters):
    brightness: float = Field(default=1.1, ge=.5, le=1.5)
    contrast: float = Field(default=1.0, ge=.5, le=1.5)
    color: float = Field(default=1.0, ge=.5, le=1.5)


class NoiseParameters(Parameters):
    std: float = Field(default=5., ge=.5, le=25.)
    seed: int = Field(default=42, ge=0, le=2**32-1)


class ResizeParameters(Parameters):
    scale: float = Field(default=.75, ge=.25, le=2.)
    resample: Literal['nearest', 'bilinear', 'bicubic', 'lanczos'] = 'lanczos'


class SharpenParameters(Parameters):
    radius: float = Field(default=2.0, ge=0., le=5.)
    percent: int = Field(default=150, ge=0, le=500)
    threshold: int = Field(default=3, ge=0, le=255)


class CropPadParameters(Parameters):
    crop_box: list[Annotated[int, Field(ge=0, le=32768)]] | None = Field(default=None, min_length=4, max_length=4)
    padding: list[Annotated[int, Field(ge=0, le=4096)]] = Field(default_factory=lambda:[0,0,0,0], min_length=4, max_length=4)
    fill_mode: Literal['constant', 'edge'] = 'constant'
    fill: list[Annotated[int, Field(ge=0, le=255)]] = Field(default_factory=lambda:[0,0,0], min_length=3, max_length=3)

    @field_validator('crop_box')
    @classmethod
    def crop_must_have_positive_area(cls, value):
        if value is not None and (value[0]>=value[2] or value[1]>=value[3]):
            raise ValueError('Crop uses [left,top,right,bottom], right/bottom exclusive, positive area required')
        return value


class PixelateParameters(Parameters):
    scale: float = Field(default=.125, ge=.01, le=1.)
    down_resample: Literal['box', 'bilinear'] = 'box'
    up_resample: Literal['nearest'] = 'nearest'
