"""Data pipeline: Scryfall bulk file -> one table row per unique illustration -> splits -> art images.

Card data and images come from Scryfall (https://scryfall.com/docs/api/bulk-data).
The held-out test sets and the label rules follow the "MTG Art Classifier" by Nathan King
(https://huggingface.co/BuildITSystems/mtg-art-cnn), so results are comparable.
"""
import gzip
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ART_DIR = DATA_DIR / "art"
BULK_FILE = RAW_DIR / "default_cards.jsonl.gz"
CARDS_CSV = DATA_DIR / "cards.csv"

# Scryfall asks every client to send these two headers
HEADERS = {"User-Agent": "manasight/0.1", "Accept": "application/json"}

SEED = 42
IMAGE_SHORT_SIDE = 224   # art is saved with its short side at 224 px (what pretrained models expect)
JPEG_QUALITY = 80

# 7 color classes: one color, no color, or two and more
COLOR_CLASSES = ["W", "U", "B", "R", "G", "Colorless", "Multicolor"]
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green",
               "Colorless": "Colorless", "Multicolor": "Multicolor"}

# 7 type classes. A card can have several types ("Artifact Creature"): the first match in this order wins.
TYPE_CLASSES = ["Creature", "Planeswalker", "Land", "Artifact", "Enchantment", "Instant", "Sorcery"]

# Test set: 10 whole sets, 2 per card-frame era (same sets as the prior work)
HELD_OUT_SETS = {
    "ice": "1993", "leg": "1993",      # Ice Age, Legends
    "7ed": "1997", "inv": "1997",      # 7th Edition, Invasion
    "lrw": "2003", "rtr": "2003",      # Lorwyn, Return to Ravnica
    "neo": "2015", "lci": "2015",      # Kamigawa: Neon Dynasty, The Lost Caverns of Ixalan
    "fra": "2024+", "dft": "2024+",    # Reality Fracture, Aetherdrift
}

# Sets left out of every split (found in notebooks/02_eda.ipynb): the pictures are not normal card paintings
JUNK_SETS = {
    "slz": "collages of card frames and text",
    "dbl": "black-and-white copies of paintings that are already in the data",
    "mb2": "test-print sketches and crops that include the card frame",
}

# Things that are not real playable cards
SKIP_LAYOUTS = {"token", "double_faced_token", "emblem", "art_series", "planar", "scheme", "vanguard"}
SKIP_SET_TYPES = {"token", "memorabilia", "funny", "minigame"}


def download_bulk_file():
    """Download Scryfall's "Default Cards" file (every English printing, about 80 MB)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if BULK_FILE.exists():
        print("already downloaded:", BULK_FILE.name)
        return
    info = requests.get("https://api.scryfall.com/bulk-data/default-cards", headers=HEADERS, timeout=30).json()
    print("downloading", info["jsonl_download_uri"])
    response = requests.get(info["jsonl_download_uri"], headers=HEADERS, timeout=300)
    response.raise_for_status()
    BULK_FILE.write_bytes(response.content)
    print(f"saved {BULK_FILE.name}: {len(response.content) / 1e6:.0f} MB, updated {info['updated_at'][:10]}")


def color_label(color_identity):
    """Color identity (list like ["R", "G"]) -> one of the 7 color classes."""
    if len(color_identity) == 0:
        return "Colorless"
    if len(color_identity) == 1:
        return color_identity[0]
    return "Multicolor"


def type_label(type_line):
    """Type line ("Legendary Artifact Creature — Golem") -> one of the 7 type classes, or None."""
    types = type_line.split("—")[0]   # the part before the dash holds the card types
    for name in TYPE_CLASSES:
        if name in types:
            return name
    return None


def is_real_card(card):
    """English paper cards only: no tokens, joke sets, test prints, oversized cards or missing images."""
    if card["lang"] != "en":
        return False
    if "paper" not in card["games"]:
        return False
    if card["oversized"]:
        return False
    if card["layout"] in SKIP_LAYOUTS:
        return False
    if card["set_type"] in SKIP_SET_TYPES:
        return False
    if card["image_status"] not in ("highres_scan", "lowres"):
        return False
    return True


def read_printings(path=BULK_FILE):
    """Read the bulk file. Returns a table with one row per printing (a card can be printed many times)."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            card = json.loads(line)
            if not is_real_card(card):
                continue

            # Two-sided cards keep their pictures inside "card_faces". We use the front face only.
            front = card
            if "image_uris" not in card and "card_faces" in card:
                front = card["card_faces"][0]
            if "illustration_id" not in front or "image_uris" not in front:
                continue

            # "Fire // Ice" style type lines: keep the front half
            type_line = card.get("type_line", front.get("type_line", "")).split(" // ")[0]

            rows.append({
                "illustration_id": front["illustration_id"],
                "oracle_id": card.get("oracle_id", front.get("oracle_id")),
                "name": card["name"],
                "set": card["set"],
                "set_name": card["set_name"],
                "released_at": card["released_at"],
                "collector_number": card["collector_number"],
                "artist": front.get("artist", card.get("artist", "")),
                "frame": card["frame"],
                "layout": card["layout"],
                "type_line": type_line,
                "color_identity": "".join(card["color_identity"]),
                "color": color_label(card["color_identity"]),
                "type": type_label(type_line),
                "art_url": front["image_uris"]["art_crop"],
            })
    return pd.DataFrame(rows)


def one_row_per_illustration(printings):
    """The same picture is often reprinted. Keep one printing per picture.

    If a picture was printed in a held-out set we keep that printing, otherwise the oldest one.
    """
    df = printings.copy()
    df["held_out"] = df["set"].isin(list(HELD_OUT_SETS))
    df = df.sort_values(["held_out", "released_at", "set", "collector_number"],
                        ascending=[False, True, True, True])
    df = df.drop_duplicates("illustration_id", keep="first")
    return df.sort_values(["released_at", "set", "collector_number"]).reset_index(drop=True)


def pick_groups(df, group_column, fractions, seed=SEED):
    """Randomly give whole groups (all pictures of one card, or of one artist) to train / val / test.

    Done separately inside every color class so each split has the same color mix.
    fractions = (val, test). Returns a Series with "train" / "val" / "test".
    """
    rng = np.random.RandomState(seed)
    val_fraction, test_fraction = fractions
    result = pd.Series("train", index=df.index)
    # count each group's pictures per color; a group goes with its most common color
    # (on a tie: the first one in COLOR_CLASSES order)
    counts = pd.crosstab(df[group_column], df["color"]).reindex(columns=COLOR_CLASSES, fill_value=0)
    group_color = counts.idxmax(axis=1)
    group_size = counts.sum(axis=1)
    for color in COLOR_CLASSES:
        groups = group_color.index[group_color == color].to_numpy()
        rng.shuffle(groups)
        sizes = group_size[groups].to_numpy()
        share = np.cumsum(sizes) / sizes.sum()   # running share of this color's pictures
        val_groups = groups[share <= val_fraction]
        test_groups = groups[(share > val_fraction) & (share <= val_fraction + test_fraction)]
        result[df[group_column].isin(val_groups)] = "val"
        result[df[group_column].isin(test_groups)] = "test"
    return result


def add_splits(df, printings, remove_by="card", drop_basics=True):
    """Add the split columns.

    split         main split: test = the 10 held-out sets, val = 12% of the other cards.
                  "removed" = other pictures of a test card (they would leak the answer), and basic lands.
    split_random  random 80/10/10 by card (ignores sets)
    split_artist  80/10/10 by artist: every artist is in one split only

    remove_by="card":    remove every other picture of a card that is in the test sets
    remove_by="picture": remove nothing (a test card's other artworks stay in training)
    drop_basics=True:    basic lands (Plains, Island, Swamp, Mountain, Forest) are left out of the main
                         split. They stay in the table and in split_random / split_artist (use is_basic).
    """
    df = df.copy()
    df["era"] = df["set"].map(HELD_OUT_SETS).fillna("")
    df["is_land"] = df["type_line"].str.split("—").str[0].str.contains("Land")
    df["is_basic"] = df["type_line"].str.contains("Basic") & df["is_land"]

    test_cards = set(printings.loc[printings["set"].isin(list(HELD_OUT_SETS)), "oracle_id"])
    is_test = df["set"].isin(list(HELD_OUT_SETS))
    is_removed = ~is_test & df["oracle_id"].isin(test_cards) & (remove_by == "card")
    if drop_basics:
        is_removed = is_removed | df["is_basic"]
        is_test = is_test & ~df["is_basic"]

    rest = df[~is_test & ~is_removed]
    df["split"] = "train"
    df.loc[rest.index, "split"] = pick_groups(rest, "oracle_id", (0.12, 0.0))
    df.loc[is_test, "split"] = "test"
    df.loc[is_removed, "split"] = "removed"

    df["split_random"] = pick_groups(df, "oracle_id", (0.10, 0.10))
    df["split_artist"] = pick_groups(df, "artist", (0.10, 0.10))
    return df


def art_path(illustration_id):
    return ART_DIR / f"{illustration_id}.jpg"


def download_one(illustration_id, art_url, tries=3):
    """Download one art crop, shrink it and save it. Returns an error message, or None if it worked.

    Connections sometimes drop, so a failed download is tried again a few times.
    """
    error_message = None
    for attempt in range(tries):
        try:
            response = requests.get(art_url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            scale = IMAGE_SHORT_SIDE / min(image.size)
            new_size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            image.save(art_path(illustration_id), "JPEG", quality=JPEG_QUALITY)
            return None
        except Exception as error:
            error_message = str(error)
            time.sleep(2)
    return error_message


def download_art(df, limit=None, workers=8):
    """Download every art crop to data/art/<illustration_id>.jpg, a few at a time (workers).

    Safe to stop and run again: pictures that are already on disk are skipped.
    Returns the list of (illustration_id, error) that failed.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm

    ART_DIR.mkdir(parents=True, exist_ok=True)
    # a picture is missing if there is no file, or an empty one (a download that was cut off)
    have = [art_path(i).exists() and art_path(i).stat().st_size > 0 for i in df["illustration_id"]]
    todo = df[[not h for h in have]]
    print(f"{len(df) - len(todo):,} of {len(df):,} already on disk")
    if limit is not None:
        todo = todo.head(limit)
    print(f"downloading {len(todo):,}")

    with ThreadPoolExecutor(workers) as pool:
        errors = list(tqdm(pool.map(download_one, todo["illustration_id"], todo["art_url"]), total=len(todo)))

    failed = [(i, e) for i, e in zip(todo["illustration_id"], errors) if e is not None]
    print(f"done, {len(failed)} failed")
    return failed


def load_cards():
    """Read data/cards.csv (made in notebooks/01_data.ipynb) and add the path of every picture.

    Pictures from JUNK_SETS get is_junk = True and are marked "removed" in every split.
    """
    df = pd.read_csv(CARDS_CSV, dtype=str, keep_default_na=False)
    df["is_land"] = df["is_land"] == "True"
    df["is_basic"] = df["is_basic"] == "True"
    df["is_junk"] = df["set"].isin(list(JUNK_SETS))
    for column in ("split", "split_random", "split_artist"):
        df.loc[df["is_junk"], column] = "removed"
    df["path"] = [str(art_path(i)) for i in df["illustration_id"]]
    return df
