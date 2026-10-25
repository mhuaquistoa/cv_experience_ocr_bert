"""Baseline 2 - Frozen sentence embeddings + Ridge (encoder is NOT fine-tuned)."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Encoder e.g. all-MiniLM-L6-v2 / bge-small-en-v1.5
    # 2. Long CVs: chunk and average embeddings
    # 3. Tune alpha on validation only
    raise NotImplementedError


if __name__ == "__main__":
    main()
