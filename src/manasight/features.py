"""Picture -> numbers. Features that later notebooks feed into simple models."""
import numpy as np


def color_histogram(thumbs, bins=4):
    """Palette of every picture as a color histogram.

    Each of R, G, B is cut into `bins` levels, which gives bins x bins x bins color boxes (4 -> 64 boxes).
    The feature is the share of the picture's pixels that falls into each box.
    It knows which colors are in the picture and how much of each, and nothing about where they are.

    thumbs: array (number of pictures, height, width, 3) with values 0-255.
    Returns an array (number of pictures, bins ** 3).
    """
    levels = (thumbs.astype(np.int64) * bins) // 256                                # 0 .. bins-1 for every channel
    box = levels[..., 0] * bins * bins + levels[..., 1] * bins + levels[..., 2]     # one box number per pixel
    box = box.reshape(len(thumbs), -1)
    histogram = np.zeros((len(thumbs), bins ** 3), dtype=np.float32)
    for i in range(len(thumbs)):
        histogram[i] = np.bincount(box[i], minlength=bins ** 3)
    return histogram / box.shape[1]
