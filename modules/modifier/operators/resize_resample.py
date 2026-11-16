import math
from PIL import Image
from shared.contracts import ModuleFailure
from modules.modifier.operators.base import ImageTransform

RESAMPLE = {'nearest':Image.Resampling.NEAREST, 'bilinear':Image.Resampling.BILINEAR,
            'bicubic':Image.Resampling.BICUBIC, 'lanczos':Image.Resampling.LANCZOS}


def apply(image, parameters, max_pixels):
    size = tuple(max(1, math.floor(dimension * parameters.scale + .5)) for dimension in image.size)
    if size[0] * size[1] > max_pixels:
        raise ModuleFailure('image_too_large', 'Resize output exceeds the configured pixel limit')
    return ImageTransform(image.resize(size, resample=RESAMPLE[parameters.resample]),
        details={'rounding':'floor(dimension * scale + 0.5), minimum 1 pixel',
                 'resample':parameters.resample, 'super_resolution':False})
