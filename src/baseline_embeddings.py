"""Baseline 2 - Frozen sentence embeddings (bge-small-en-v1.5) + Ridge. The encoder is NOT fine-tuned.

Long CVs: the encoder reads at most 512 tokens, so each CV is split into windows of CHUNK_WORDS
words, every window is embedded, and the CV vector is the mean of its window vectors. Nothing is
truncated away, at the cost of blurring the order between windows.
"""
import time

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from utils import (EVAL_SETS, MAIN_VARIANT, MODELS, TARGET, TEXT_VARIANTS, clip_years, ensure_dirs, load_sets,
                   save_predictions, set_seed)

ENCODER = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS = 250  # ~350 tokens, safely under the 512-token limit
ALPHAS = [0.1, 0.3, 1, 3, 10, 30]


def chunks(text: str) -> list:
    words = text.split()
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words), 1), CHUNK_WORDS)] or [""]


def embed(encoder, texts) -> np.ndarray:
    pieces, owner = [], []
    for i, t in enumerate(texts):
        for c in chunks(t):
            pieces.append(c)
            owner.append(i)
    vecs = encoder.encode(pieces, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    owner = np.array(owner)
    out = np.stack([vecs[owner == i].mean(axis=0) for i in range(len(texts))])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def load_encoder():
    return SentenceTransformer(ENCODER, device="cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    set_seed()
    ensure_dirs()
    encoder = load_encoder()
    for variant in TEXT_VARIANTS:
        sets = load_sets(variant)
        t0 = time.perf_counter()
        X = {name: embed(encoder, sets[name]["text"].tolist()) for name in ["train", "val"]}
        y_tr, y_va = sets["train"][TARGET].values, sets["val"][TARGET].values
        scores = {a: mean_absolute_error(y_va, clip_years(Ridge(alpha=a).fit(X["train"], y_tr).predict(X["val"])))
                  for a in ALPHAS}
        best = min(scores, key=scores.get)
        model = Ridge(alpha=best).fit(X["train"], y_tr)
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = {name: model.predict(embed(encoder, sets[name]["text"].tolist()))
                 for name in EVAL_SETS}
        infer = (time.perf_counter() - t0) / sum(len(sets[n]) for n in preds)
        save_predictions("embeddings", variant, sets, preds,
                         {"train_seconds": train_s, "infer_seconds_per_cv": infer, "alpha": best,
                          "encoder": ENCODER, "val_mae_by_alpha": scores})
        print(f"embeddings [{variant}] alpha={best} val MAE={scores[best]:.2f}")
        if variant == MAIN_VARIANT:
            joblib.dump(model, MODELS / "baselines" / "embeddings_ridge.joblib")


if __name__ == "__main__":
    main()
