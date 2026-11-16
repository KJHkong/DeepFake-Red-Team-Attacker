"""Source: https://numpy.org/doc/2.1/reference/random/bit_generators/pcg64.html"""
import numpy as np
from PIL import Image
from modules.modifier.operators.base import ImageTransform


def apply(image, parameters, max_pixels):
    pixels = np.asarray(image, dtype=np.float32)
    rng = np.random.Generator(np.random.PCG64(parameters.seed))
    noise = rng.normal(0., parameters.std, pixels.shape).astype(np.float32)
    output = np.rint(np.clip(pixels + noise, 0, 255)).astype(np.uint8)
    return ImageTransform(Image.fromarray(output), details={'rng':'numpy.Generator(PCG64)',
        'noise_distribution':'independent normal per RGB channel', 'std_unit':'8-bit pixel values [0,255]',
        'rounding':'numpy.rint, ties to even; clip to [0,255]', 'numpy_version':np.__version__})
