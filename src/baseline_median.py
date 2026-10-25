"""Baseline 0 - Always predict the training-set median (the floor every model must beat)."""
import time

import numpy as np

from utils import EVAL_SETS, MODELS, TARGET, TEXT_VARIANTS, ensure_dirs, load_sets, save_json, save_predictions


def main() -> None:
    ensure_dirs()
    for variant in TEXT_VARIANTS:  # text-independent, saved per variant to keep tables aligned
        sets = load_sets(variant)
        t0 = time.perf_counter()
        median = float(np.median(sets["train"][TARGET]))
        train_s = time.perf_counter() - t0
        preds = {name: np.full(len(sets[name]), median) for name in EVAL_SETS}
        save_predictions("median", variant, sets, preds,
                         {"train_seconds": train_s, "infer_seconds_per_cv": 0.0})
    save_json({"median": median}, MODELS / "baselines" / "median.json")
    print(f"Median of train = {median}")


if __name__ == "__main__":
    main()
