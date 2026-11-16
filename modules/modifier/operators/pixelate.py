"""Controlled pixelation: bounded downsample, nearest upsample to original canvas."""
import math
from PIL import Image
from modules.modifier.operators.base import ImageTransform


def apply(image,parameters,max_pixels):
    down=tuple(max(1,math.floor(value*parameters.scale+.5)) for value in image.size)
    resample={'box':Image.Resampling.BOX,'bilinear':Image.Resampling.BILINEAR}[parameters.down_resample]
    output=image.resize(down,resample=resample).resize(image.size,resample=Image.Resampling.NEAREST)
    return ImageTransform(output,details=dict(method='downsample_then_nearest_upsample',scale=parameters.scale,
        downsample_size=list(down),down_resample=parameters.down_resample,up_resample='nearest',
        output_size=list(image.size),rounding='max(1,floor(dimension*scale+0.5))',
        purpose='controlled_damage_robustness',automatic_quality_improvement=False))
