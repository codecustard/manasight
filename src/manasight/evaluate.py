"""Scores used in every notebook, so all models are measured the same way."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def score(true_labels, probabilities, classes):
    """Accuracy, macro-F1 and top-2 accuracy.

    true_labels:   the right class name of every picture
    probabilities: array (number of pictures, number of classes), one column per class in the order of `classes`
    """
    classes = np.array(classes)
    true_labels = np.array(true_labels)
    predicted = classes[probabilities.argmax(axis=1)]
    two_best = classes[np.argsort(-probabilities, axis=1)[:, :2]]        # the two most likely classes
    in_two_best = (two_best == true_labels[:, None]).any(axis=1)
    return {"accuracy": accuracy_score(true_labels, predicted),
            "macro-F1": f1_score(true_labels, predicted, average="macro", labels=list(classes), zero_division=0),
            "top-2": float(in_two_best.mean())}
