"""Pillow UnsharpMask; sharpening may amplify artifacts, not a quality guarantee."""
from PIL import ImageFilter
from modules.modifier.operators.base import ImageTransform


def apply(image, parameters, max_pixels):
    output=image.filter(ImageFilter.UnsharpMask(radius=parameters.radius,
        percent=parameters.percent,threshold=parameters.threshold))
    return ImageTransform(output,details={'method':'Pillow.ImageFilter.UnsharpMask',
        'radius_unit':'pixels','percent_unit':'percent','threshold_unit':'8bit brightness difference',
        'quality_improvement':'not_evaluated','may_amplify_artifacts':True})
