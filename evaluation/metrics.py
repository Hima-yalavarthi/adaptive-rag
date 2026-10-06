"""Answer and evidence metrics, following the official HotpotQA normalization."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_norm, gold_norm = normalize_answer(prediction), normalize_answer(gold)

    # yes/no/noanswer must match exactly, as in the official HotpotQA script.
    special = {"yes", "no", "noanswer"}
    if (pred_norm in special or gold_norm in special) and pred_norm != gold_norm:
        return 0.0

    pred_tokens, gold_tokens = pred_norm.split(), gold_norm.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def set_precision_recall(predicted: set, gold: set) -> tuple[float, float]:
    """Precision and recall of a predicted set against a gold set (0.0 when empty)."""
    hits = len(predicted & gold)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold) if gold else 0.0
    return precision, recall
