"""Release wrappers for the historical scalar probability fusion.

These functions do not make cross-fold selection fully nested. Keep every
selection step inside an outer training set for future confirmatory studies.
"""
import numpy as np
from sklearn.metrics import f1_score
from .executed_components import adjusted


def probabilities(value):
    p = np.asarray(value, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite (n, 3) diagnosis probabilities")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Rows must be nonnegative and sum to one")
    return p


def fuse(tree, neural, alpha):
    tree, neural = probabilities(tree), probabilities(neural)
    if tree.shape != neural.shape or not 0 <= alpha <= 1:
        raise ValueError("Aligned predictions and alpha in [0,1] required")
    return alpha * tree + (1 - alpha) * neural


def select_alpha_tau(y, tree, neural, patterns, priors):
    """Historical search grid on a supplied tuning set, with first-max tie rule.

    The caller is responsible for subject separation and tuning-set provenance.
    The frozen paper used other-fold predictions in a NONNESTED selection chain.
    """
    tree, neural = probabilities(tree), probabilities(neural)
    y, patterns = np.asarray(y), np.asarray(patterns, dtype=int)
    if tree.shape != neural.shape or len(y) != len(tree) or len(patterns) != len(y):
        raise ValueError("Tuning arrays must align")
    priors = np.asarray(priors, dtype=float)
    if priors.shape != (16, 3) or not np.isfinite(priors).all() or np.any(priors < 0):
        raise ValueError("Expected finite nonnegative (16,3) pattern priors")
    if np.any((patterns < 1) | (patterns > 15)):
        raise ValueError("Patterns must be nonempty four-bit codes")
    best = (-1.0, 1.0, 0.0)
    for alpha in np.round(np.arange(0, 1.0001, .05), 2):
        mixture = fuse(tree, neural, float(alpha))
        for tau in [0., .25, .5, .75, 1., 1.5]:
            prediction = adjusted(mixture, patterns, priors, tau)
            score = f1_score(y, prediction.argmax(1), labels=[0,1,2], average="macro", zero_division=0)
            if score > best[0]:
                best = (float(score), float(alpha), float(tau))
    return {"tuning_macro_f1":best[0], "alpha":best[1], "tau":best[2]}
