"""Baseline 1 - TF-IDF (word 1-2 grams, optional char n-grams) + Ridge."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Decide whether to keep digits/years as tokens (justify)
    # 2. Tune alpha on validation only; save model to models/baselines/
    raise NotImplementedError


if __name__ == "__main__":
    main()
