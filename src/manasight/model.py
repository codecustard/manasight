"""Models that are trained on top of the saved CLIP features."""
import numpy as np
import torch
from torch import nn


class SmallNetwork(nn.Module):
    """Two layers: CLIP features -> hidden layer -> classes.

    The hidden layer lets the model combine features ("fire AND no water"), which a single layer cannot do.
    Dropout switches off random hidden units during training so the model does not lean on a few of them.
    """

    def __init__(self, inputs=512, hidden=256, classes=7, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, x):
        return self.layers(x)


def predict_probabilities(model, x):
    """Class probabilities for a numpy array of features: array (number of pictures, number of classes)."""
    model.eval()
    with torch.inference_mode():
        scores = model(torch.from_numpy(x))
        probabilities = torch.softmax(scores, dim=1)
    return probabilities.numpy()


def count_weights(model):
    total = 0
    for parameter in model.parameters():
        total += parameter.numel()
    return total
