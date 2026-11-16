"""Source: https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html"""
from PIL import ImageEnhance
from modules.modifier.operators.base import ImageTransform


def apply(image, parameters, max_pixels):
    output = ImageEnhance.Brightness(image).enhance(parameters.brightness)
    output = ImageEnhance.Contrast(output).enhance(parameters.contrast)
    output = ImageEnhance.Color(output).enhance(parameters.color)
    return ImageTransform(output, details={'operation_order':['brightness','contrast','color']})
