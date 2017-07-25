import numpy as np
from scipy.special import expit  # sigmoid

from sklearn.metrics import f1_score


def f1_micro_score(y_true, y_pred):
    score = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='micro')
    return score


def dice_score(y_true, y_pred):
    epsilon = 1e-5
    denominator = y_true.sum(axis=(1, 2)) + y_pred.sum(axis=(1, 2)) + epsilon
    numerator = 2 * np.sum(y_true * y_pred, axis=(1, 2)) + epsilon

    dice = numerator / denominator

    dice = dice.mean()

    return dice


def topcoder_metric(y_true, y_pred):
    f1 = f1_micro_score(y_true, y_pred)
    d = dice_score(y_true, y_pred)
    score = 1000000.0 * (f1 + d) / 2.0

    return score

    # y_pred_probs = expit(y_pred_logits)
    # y_pred = np.zeros_like(y_pred_probs)
    # y_pred[y_pred_probs>=threshold] = 1
