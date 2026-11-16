"""Helpers for looking at the data: a tiny version of every picture, kept in one array."""
import numpy as np
from PIL import Image

from manasight.data import DATA_DIR

THUMB_SIZE = 32
THUMBS_FILE = DATA_DIR / f"thumbs_{THUMB_SIZE}.npz"


def make_thumbnail(path, size=THUMB_SIZE):
    """Open one picture. Returns it squashed to size x size pixels (RGB, 0-255) and its real (width, height)."""
    with Image.open(path) as image:
        real_size = image.size
        small = image.convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(small, dtype=np.uint8), real_size


def load_thumbnails(cards):
    """Tiny version of every picture, in the same order as `cards`.

    Returns thumbs, shape (number of pictures, 32, 32, 3), and sizes, shape (number of pictures, 2) = width, height.
    Building them opens every image once (a few minutes). The result is saved in data/ and reused next time.
    """
    from tqdm.auto import tqdm

    ids = cards["illustration_id"].to_numpy().astype(str)
    if THUMBS_FILE.exists():
        saved = np.load(THUMBS_FILE)
        if len(saved["ids"]) == len(ids) and (saved["ids"] == ids).all():
            return saved["thumbs"], saved["sizes"]

    thumbs = np.zeros((len(cards), THUMB_SIZE, THUMB_SIZE, 3), dtype=np.uint8)
    sizes = np.zeros((len(cards), 2), dtype=np.int64)
    for i, path in enumerate(tqdm(cards["path"])):
        thumbs[i], sizes[i] = make_thumbnail(path)
    np.savez(THUMBS_FILE, ids=ids, thumbs=thumbs, sizes=sizes)
    return thumbs, sizes


def brightness_and_saturation(thumbs):
    """Two numbers per picture, both from 0 to 1.

    brightness: 0 = black, 1 = full light.   saturation: 0 = grey, 1 = pure vivid color.
    (These are the V and S of the HSV color model, averaged over the pixels.)
    """
    pixels = thumbs.astype(np.float32) / 255
    highest = pixels.max(axis=3)   # the strongest of R, G, B in each pixel
    lowest = pixels.min(axis=3)
    brightness = highest.mean(axis=(1, 2))
    saturation = ((highest - lowest) / np.maximum(highest, 1e-6)).mean(axis=(1, 2))
    return brightness, saturation


def gray_fingerprints(thumbs):
    """A 256-number fingerprint per picture: its 16x16 grey version, with the average removed and length 1.

    The dot product of two fingerprints is their similarity: 1 = same picture, 0 = unrelated.
    Color and overall brightness do not matter, so a recolored or darker copy still matches.
    """
    gray = thumbs.astype(np.float32).mean(axis=3)                       # (n, 32, 32)
    gray = gray.reshape(len(gray), 16, 2, 16, 2).mean(axis=(2, 4))      # (n, 16, 16): average 2x2 blocks
    gray = gray.reshape(len(gray), -1)
    gray = gray - gray.mean(axis=1, keepdims=True)
    length = np.linalg.norm(gray, axis=1, keepdims=True)
    return gray / np.maximum(length, 1e-6)
