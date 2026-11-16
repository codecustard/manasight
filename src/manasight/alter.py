"""Altered versions of a picture, to test what a model relies on.

to_grayscale   removes the colors and keeps the content
shuffle_tiles  keeps the colors and breaks up the content
"""
import numpy as np
from PIL import Image
from torchvision.transforms import functional

SIZE = 224


def to_square(image):
    """The 224x224 picture a model sees: short side resized to 224, then the center cut out (exactly as CLIP does)."""
    image = image.convert("RGB")
    image = functional.resize(image, SIZE, interpolation=functional.InterpolationMode.BICUBIC)
    return functional.center_crop(image, SIZE)


def to_grayscale(image):
    """Same picture without color: every pixel becomes a shade of grey. Shapes and brightness stay."""
    return image.convert("L").convert("RGB")


def shuffle_tiles(image, tiles, seed=0):
    """Cut the 224x224 picture into tiles x tiles squares and put them back in a random order.

    Every pixel is still there, so the palette is unchanged. The more tiles, the less of the content is left:
    2 gives four large pieces, 224 moves every single pixel.
    """
    pixels = np.asarray(image)
    side = SIZE // tiles
    pieces = []
    for row in range(tiles):
        for column in range(tiles):
            pieces.append(pixels[row * side:(row + 1) * side, column * side:(column + 1) * side])
    order = np.random.RandomState(seed).permutation(len(pieces))
    shuffled = np.zeros_like(pixels)
    position = 0
    for row in range(tiles):
        for column in range(tiles):
            shuffled[row * side:(row + 1) * side, column * side:(column + 1) * side] = pieces[order[position]]
            position += 1
    return Image.fromarray(shuffled)
