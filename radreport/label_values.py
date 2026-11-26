"""Parse Stanford CheXpert NLP CSV labels safely.

Official labeler typically emits 1.0 (positive), 0.0 (negative),
-1.0 (uncertain), and empty (unmentioned).

Unknown, missing and uncertain are deliberately not converted to negative.
"""
import math


def parse_binary_chexpert(raw) -> int | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(number):
        return None
    if number == 1.0:
        return 1
    if number == 0.0:
        return 0
    return None  # Uncertain -1.0, NaN, unexpected codes: mask out.
