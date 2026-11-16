"""JPEG encoding options: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg"""
from modules.modifier.operators.base import ImageTransform


def apply(image, parameters, max_pixels):
    return ImageTransform(image.copy(), format='JPEG', encode_options={
        'quality':parameters.quality, 'subsampling':parameters.subsampling,
        'optimize':False, 'progressive':False}, details={'encoding':'RGB JPEG, explicit quality and subsampling'})
