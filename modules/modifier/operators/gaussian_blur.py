"""Pillow radius is Gaussian standard deviation, in pixels.
Source: https://github.com/python-pillow/Pillow/blob/10.2.0/src/PIL/ImageFilter.py
"""
from PIL import ImageFilter
from modules.modifier.operators.base import ImageTransform


def apply(image, parameters, max_pixels):
    return ImageTransform(image.filter(ImageFilter.GaussianBlur(parameters.radius)),
        details={'implementation':'Pillow GaussianBlur', 'radius_unit':'pixels'})
