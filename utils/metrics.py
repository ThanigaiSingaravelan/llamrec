# utils/metrics.py
from typing import List
import numpy as np


def precision_at_k(preds: List[str], actuals: List[str], k: int = 3) -> float:
    preds = [p.lower().strip() for p in preds[:k]]
    actuals = set(a.lower().strip() for a in actuals)
    return sum(p in actuals for p in preds) / k


def recall_at_k(preds: List[str], actuals: List[str], k: int = 3) -> float:
    preds = [p.lower().strip() for p in preds[:k]]
    actuals = set(a.lower().strip() for a in actuals)
    return sum(p in actuals for p in preds) / len(actuals) if actuals else 0.0


def ndcg_at_k(preds: List[str], actuals: List[str], k: int = 3) -> float:
    actuals = set(a.lower().strip() for a in actuals)
    dcg = 0.0
    for i, p in enumerate(preds[:k]):
        if p.lower().strip() in actuals:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actuals), k)))
    return dcg / idcg if idcg else 0.0


def jaccard_diversity(items: List[str]) -> float:
    if len(items) < 2:
        return 0.0
    items = [set(i.lower().split()) for i in items]
    diversities = [1 - len(a & b) / len(a | b) for i, a in enumerate(items) for b in items[i + 1:]]
    return float(np.mean(diversities)) if diversities else 0.0
