"""CLIP: a pretrained model that turns pictures and sentences into the same kind of numbers.

A picture and a sentence that describe the same thing get similar numbers ("features", 512 per item).
All features here have length 1, so the dot product of two of them is their similarity (higher = closer).
"""
from pathlib import Path

import numpy as np
from PIL import Image

from manasight.data import DATA_DIR

MODEL_NAME = "ViT-B-32-quickgelu"     # CLIP ViT-B/32 ...
WEIGHTS = "openai"                    # ... with OpenAI's original weights
IMAGE_FEATURES_FILE = DATA_DIR / "clip_image_features.npz"
WORD_FEATURES_FILE = DATA_DIR / "clip_word_features.npz"
PARTS_DIR = DATA_DIR / "clip_parts"

# 13,000 ordinary English words, most common first: the words CLIP has in its own vocabulary that are also
# in an English dictionary (Webster's Second, lower-case entries), without stop words like "the".
# Nothing in this list was chosen with Magic in mind.
WORDS_FILE = Path(__file__).parent / "words.txt"


def load_clip():
    """Load CLIP (downloads about 350 MB the first time). Returns model, image transform, tokenizer."""
    import open_clip

    model, _, transform = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=WEIGHTS)
    model.eval()   # we never train CLIP itself
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, transform, tokenizer


def image_features(cards, model, transform, chunk=1000, batch=50):
    """CLIP features of every picture in `cards`: array (number of pictures, 512), same order as `cards`.

    Slow the first time (about 20 pictures per second on a CPU). The work is saved in chunks of 1,000 pictures,
    so it is safe to stop and run again. The finished result is saved in data/ and reused next time.
    """
    import torch
    from tqdm.auto import tqdm

    ids = cards["illustration_id"].to_numpy().astype(str)
    if IMAGE_FEATURES_FILE.exists():
        saved = np.load(IMAGE_FEATURES_FILE)
        if len(saved["ids"]) == len(ids) and (saved["ids"] == ids).all():
            return saved["features"]

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = cards["path"].to_numpy()
    parts = []
    for start in tqdm(range(0, len(cards), chunk)):
        part_file = PARTS_DIR / f"part_{start:06d}_{ids[start][:8]}.npy"
        if part_file.exists():
            parts.append(np.load(part_file))
            continue
        chunk_features = []
        chunk_paths = paths[start:start + chunk]
        for first in range(0, len(chunk_paths), batch):
            images = []
            for path in chunk_paths[first:first + batch]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))   # resize, center crop to 224x224, normalize
            with torch.no_grad():
                features = model.encode_image(torch.stack(images))
            features = features / features.norm(dim=1, keepdim=True)  # length 1
            chunk_features.append(features.numpy().astype(np.float32))
        part = np.concatenate(chunk_features)
        np.save(part_file, part)
        parts.append(part)

    features = np.concatenate(parts)
    np.savez(IMAGE_FEATURES_FILE, ids=ids, features=features)
    return features


def text_features(sentences, model, tokenizer, batch=250):
    """CLIP features of a list of sentences: array (number of sentences, 512)."""
    import torch
    from tqdm.auto import tqdm

    parts = []
    starts = range(0, len(sentences), batch)
    if len(sentences) > 2000:
        starts = tqdm(starts)   # long lists take minutes: show progress
    for first in starts:
        tokens = tokenizer(list(sentences[first:first + batch]))
        with torch.no_grad():
            features = model.encode_text(tokens)
        features = features / features.norm(dim=1, keepdim=True)
        parts.append(features.numpy().astype(np.float32))
    return np.concatenate(parts)


def word_features(model, tokenizer):
    """The neutral word list and the CLIP features of "a painting of <word>" for every word.

    Returns words (list) and features (number of words, 512). Takes a few minutes once, then loads from data/.
    """
    words = WORDS_FILE.read_text().split()
    if WORD_FEATURES_FILE.exists():
        saved = np.load(WORD_FEATURES_FILE)
        if len(saved["words"]) == len(words) and (saved["words"] == np.array(words)).all():
            return words, saved["features"]
    features = text_features([f"a painting of {word}" for word in words], model, tokenizer)
    np.savez(WORD_FEATURES_FILE, words=np.array(words), features=features)
    return words, features


def average_feature(features):
    """Average several features into one, and give it length 1 again."""
    average = features.mean(axis=0)
    return average / np.linalg.norm(average)


def class_features(prompts, classes, model, tokenizer):
    """One text feature per class. prompts = {class: [sentence, sentence, ...]}; several sentences are averaged."""
    rows = []
    for name in classes:
        rows.append(average_feature(text_features(prompts[name], model, tokenizer)))
    return np.stack(rows)


def zero_shot(picture_features, class_text_features):
    """Probability of every class for every picture, from similarity alone (no training).

    The similarities are multiplied by 100 before the softmax, as CLIP itself does.
    """
    scores = 100 * picture_features @ class_text_features.T
    scores = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def discover_words(picture_features, labels, classes, words, all_word_features, how_many):
    """For every class, the words that fit its pictures much better than the other classes' pictures.

    Score of a word for a class = its average similarity to that class's pictures
                                  minus its average similarity to all other pictures.
    Returns {class: [best word, second best, ...]}.
    """
    labels = np.array(labels)
    found = {}
    for name in classes:
        inside = picture_features[labels == name].mean(axis=0)
        outside = picture_features[labels != name].mean(axis=0)
        gap = all_word_features @ (inside - outside)
        best = np.argsort(-gap)[:how_many]
        found[name] = [words[i] for i in best]
    return found


def features_of_words(chosen, words, all_word_features):
    """One class feature from a list of chosen words (the average of their features)."""
    position = {word: i for i, word in enumerate(words)}
    rows = [position[word] for word in chosen]
    return average_feature(all_word_features[rows])
