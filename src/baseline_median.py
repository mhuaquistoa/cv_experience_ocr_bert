"""Baseline 0 - Always predict the training-set median."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # Fit on train, predict on val/test, save predictions to output/metrics/
    raise NotImplementedError


if __name__ == "__main__":
    main()
