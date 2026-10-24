"""Bonus baseline - rule-based date arithmetic: extract every job date range with a regex, merge
overlapping intervals and add up the covered years. No training; CVs without any detected range
fall back to the training median. An honest test of whether BERT beats simple arithmetic.
"""
import time

import numpy as np

from utils import DATE_RANGE, EVAL_SETS, TARGET, TEXT_VARIANTS, ensure_dirs, load_sets, save_predictions


def career_years(text: str):
    spans = []
    for y0, m0, y1, m1 in DATE_RANGE.findall(text):
        start, end = int(y0) + (int(m0) - 1) / 12, int(y1) + (int(m1) - 1) / 12
        if end > start:
            spans.append((start, end))
    if not spans:
        return None
    total, cur_s, cur_e = 0.0, *sorted(spans)[0]
    for s, e in sorted(spans)[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + cur_e - cur_s


def main() -> None:
    ensure_dirs()
    for variant in TEXT_VARIANTS:
        sets = load_sets(variant)
        fallback = float(np.median(sets["train"][TARGET]))
        t0 = time.perf_counter()
        preds = {}
        for name in EVAL_SETS:
            years = [career_years(t) for t in sets[name]["text"]]
            preds[name] = np.array([fallback if y is None else y for y in years])
        infer = (time.perf_counter() - t0) / sum(len(sets[n]) for n in EVAL_SETS)
        found = np.mean([career_years(t) is not None for t in sets["test"]["text"]])
        save_predictions("rules", variant, sets, preds, {"train_seconds": 0.0, "infer_seconds_per_cv": infer,
                                                         "share_test_cvs_with_dates": found})
        mae = np.mean(np.abs(np.clip(preds["val"], 0, 20) - sets["val"][TARGET].values))
        print(f"rules [{variant}] val MAE={mae:.3f}, test CVs with dates found: {found:.1%}")


if __name__ == "__main__":
    main()
