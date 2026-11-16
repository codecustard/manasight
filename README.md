# manasight: art2mana

Can a model tell a Magic: The Gathering card's **color** and **type** from its artwork alone?
And if it can, is it reading the *palette* (red = fire tones) or the *content* (goblins, swamps, angels)?

Work in progress. Built with PyTorch; a Gradio app comes at the end.

## Prior work

manasight builds on Nathan King's [MTG Art Classifier](https://huggingface.co/BuildITSystems/mtg-art-cnn)
([demo](https://huggingface.co/spaces/BuildITSystems/mtg-art-demo), October 2026), which trained a small CNN from scratch
and compared it with a plain color histogram: 36.8% vs 34.4% accuracy on color. That result, a network landing close
to the palette alone, is the starting point here.

The follow-up question: does a **pretrained** model, which already knows what objects look like, get further past the
histogram? To keep the numbers comparable, manasight follows the same evaluation design: the same 10 held-out test sets,
the same 7 color classes and the same 7 type classes. The code is written from scratch and the split is rebuilt from
the published description, so the counts are close (within 0.3%) but not identical.

## Dataset

Card data and art from [Scryfall](https://scryfall.com/docs/api/bulk-data), built in [01_data](notebooks/01_data.ipynb):

- 48,278 unique illustrations (English paper cards; no tokens, joke sets or test prints; front picture only)
- **Color** (7 classes, from color identity): White, Blue, Black, Red, Green, Colorless, Multicolor
- **Type** (7 classes, first match wins): Creature > Planeswalker > Land > Artifact > Enchantment > Instant > Sorcery
- **Main split:** test = 10 whole held-out sets, 2 per card-frame era (Ice Age, Legends, Seventh Edition, Invasion,
  Lorwyn, Return to Ravnica, Kamigawa: Neon Dynasty, The Lost Caverns of Ixalan, Aetherdrift, Reality Fracture).
  Other paintings of the test cards, and basic lands, are removed. Train 36,222 / val 4,935 / test 3,392.
- Extra splits for later questions: random by card, and by artist

- Three sets are left out of every split because their pictures are not normal card paintings (black-and-white
  copies, test-print sketches, collages): 1,075 illustrations, found in [02_eda](notebooks/02_eda.ipynb).
  That leaves train 35,383 / val 4,798 / test 3,392. The test set is not affected.

`data/cards.csv` (one row per illustration: labels, splits, artist, image address) is in the repo.
The images (about 1 GB) are downloaded by the notebook.

## Results so far

Test set (10 held-out sets), color, 7 classes:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Chance | 14.3% | |
| Always "Multicolor" | 20.2% | 0.048 |
| Color histogram + logistic regression ([03](notebooks/03_histogram.ipynb)) | 34.0% | 0.304 |
| CLIP zero-shot, hand-written content prompts ([04](notebooks/04_clip_zero_shot.ipynb)) | 27.9% | 0.280 |
| CLIP zero-shot, palette prompts | 32.4% | 0.307 |
| CLIP zero-shot, words discovered from the training art | 36.2% | 0.335 |

The palette alone gets a third of the cards right. It finds Green, Red and Blue about half the time, almost never
finds Colorless, and drops to 22% on the oldest sets.

Type, 7 classes: always "Creature" scores 49.9% accuracy and 0.095 macro-F1. The color histogram does no better
(49.8%, 0.095): the palette says nothing about type.
CLIP zero-shot reaches 0.284 macro-F1 on type with no training: content is visible where the palette is blind.

Without any guidance, CLIP describes the colors by itself: Black as "ghoul, zombie, undead", Red as "lava, inferno,
fireball", Colorless as "machinery, spaceship". Those discovered words beat our hand-written prompts and the palette.
CLIP finds Black, Red and Colorless far better than the histogram; the histogram still wins on Blue and Green.

## Plan

Each notebook tests one hypothesis, and its result sets up the next.

| # | Notebook | Question |
|---|---|---|
| 01 | [data](notebooks/01_data.ipynb) | Build the dataset and the splits |
| 02 | [EDA](notebooks/02_eda.ipynb) | What do the pictures look like? Chance and majority baselines |
| 03 | [Color histogram](notebooks/03_histogram.ipynb) | How far does the palette alone get? |
| 04 | [CLIP zero-shot](notebooks/04_clip_zero_shot.ipynb) | Does CLIP know the game's themes without training? |
| 05 | CLIP, trained | Does a trained content model beat the histogram by a lot? |
| 06 | Grayscale and shuffled tiles | Content or palette? |
| 07 | Artist and era | Is part of it artist style? Did the look of colors change over time? |
| 08 | Multi-label color | Does multicolor art blend its colors' cues? |
| 09 | Type | Which types are visible in art? Human baseline |
| 10 | Grad-CAM and errors | What does the model look at? |

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data.ipynb
```

## Credits

- Card data and images: [Scryfall](https://scryfall.com)
- Evaluation design (held-out sets, label rules): Nathan King, [MTG Art Classifier](https://huggingface.co/BuildITSystems/mtg-art-cnn)
- Card art is © Wizards of the Coast and the individual artists; `data/cards.csv` names the artist of every illustration

manasight is unofficial Fan Content permitted under the Fan Content Policy. Not approved/endorsed by Wizards.
Portions of the materials used are property of Wizards of the Coast. © Wizards of the Coast LLC.
