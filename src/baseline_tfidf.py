"""Baseline 1 - TF-IDF (word 1-2 grams + char 3-5 grams) + Ridge.

Digits and years are KEPT as tokens by default: job dates are the evidence for experience, so
removing them would hide the signal. The alternative (--no-digits) is reported in Question 3.
Ridge alpha is tuned on the validation set only.
Usage: python src/baseline_tfidf.py [--no-digits]
"""
import argparse
import re
import time

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (EVAL_SETS, MAIN_VARIANT, MODELS, TARGET, TEXT_VARIANTS, clip_years, ensure_dirs,
                   load_sets, save_predictions)

ALPHAS = [0.1, 0.3, 1, 3, 10, 30]


def strip_digits(text: str) -> str:
    return re.sub(r"\d", " ", text.lower())


def build(alpha: float, keep_digits: bool = True):
    pre = None if keep_digits else strip_digits
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True,
                        preprocessor=pre),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50000,
                        sublinear_tf=True, preprocessor=pre),
    )
    return make_pipeline(features, Ridge(alpha=alpha))


def fit_variant(variant: str, keep_digits: bool) -> None:
    sets = load_sets(variant)
    tr, va = sets["train"], sets["val"]
    t0 = time.perf_counter()
    scores = {}
    for a in ALPHAS:
        m = build(a, keep_digits).fit(tr["text"], tr[TARGET])
        scores[a] = mean_absolute_error(va[TARGET], clip_years(m.predict(va["text"])))
    best = min(scores, key=scores.get)
    model = build(best, keep_digits).fit(tr["text"], tr[TARGET])
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = {name: model.predict(sets[name]["text"]) for name in EVAL_SETS}
    infer = (time.perf_counter() - t0) / sum(len(sets[n]) for n in preds)
    name = "tfidf" if keep_digits else "tfidf_nodigits"
    save_predictions(name, variant, sets, preds, {"train_seconds": train_s,
                                                  "infer_seconds_per_cv": infer, "alpha": best,
                                                  "val_mae_by_alpha": scores})
    print(f"{name} [{variant}] alpha={best} val MAE={scores[best]:.2f}")
    if variant == MAIN_VARIANT:
        joblib.dump(model, MODELS / "baselines" / f"{name}.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-digits", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    for variant in TEXT_VARIANTS:
        fit_variant(variant, keep_digits=not args.no_digits)


if __name__ == "__main__":
    main()
