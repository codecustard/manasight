"""Fine-tuning: train CLIP's own image network on the card art, not only a classifier on top of it."""
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

# CLIP was trained with these values to normalize the three color channels
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def train_transform():
    """Training pictures: a random crop (60% to 100% of the picture) and a random left-right flip.

    Every epoch sees a slightly different version of each picture, which makes memorizing harder.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class ArtDataset(Dataset):
    """Card art for a DataLoader: gives (picture as a tensor, class number)."""

    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            picture = self.transform(image.convert("RGB"))
        return picture, self.labels[index]


class ClipClassifier(nn.Module):
    """CLIP's image network with a new last layer for our classes.

    visual: the image half of CLIP (picture -> 512 numbers)
    head:   new layer (512 numbers -> one score per class)
    """

    def __init__(self, visual, classes):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(512, classes)

    def forward(self, x):
        return self.head(self.visual(x))


def set_clip_trainable(model, trainable):
    """Freeze (False) or unfreeze (True) CLIP's own weights. The new last layer always stays trainable."""
    for parameter in model.visual.parameters():
        parameter.requires_grad = trainable


def predict(model, loader, device):
    """Class probabilities for every picture of a DataLoader: array (number of pictures, number of classes)."""
    model.eval()
    parts = []
    with torch.inference_mode():
        for pictures, labels in loader:
            scores = model(pictures.to(device))
            parts.append(torch.softmax(scores, dim=1).cpu().numpy())
    return np.concatenate(parts)
