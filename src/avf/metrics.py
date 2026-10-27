"""Visit-level metrics with participant-based support for diagnosis-pattern cells."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from .fusion import probabilities


def supported_cells(y, patterns, participants, min_subjects=20):
    frame = pd.DataFrame({"y":y, "pattern":patterns, "participant":participants})
    if frame.isna().any().any() or min_subjects < 1:
        raise ValueError("Missing labels/groups or invalid support threshold")
    counts = frame.groupby(["pattern","y"]).participant.nunique()
    return {(int(p),int(c)) for (p,c),n in counts.items() if n >= min_subjects}


def evaluate(y, p, patterns, cells):
    y, p, patterns = np.asarray(y), probabilities(p), np.asarray(patterns)
    if len(y) != len(p) or len(patterns) != len(y) or set(np.unique(y)) != {0,1,2}:
        raise ValueError("Aligned arrays containing all three diagnoses required")
    pred = p.argmax(1)
    recall = []
    for pattern, diagnosis in sorted(cells):
        use = (patterns == pattern) & (y == diagnosis)
        if not use.any():
            raise ValueError("A fixed support cell has no evaluation visits")
        recall.append(float((pred[use] == diagnosis).mean()))
    return {"macro_auroc":float(np.mean([roc_auc_score(y == c,p[:,c]) for c in range(3)])),
            "macro_f1":float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "supported_cell_recall":float(np.mean(recall)) if recall else float("nan")}
