"""Stage 2a - OCR quality: CER / WER vs the reference text and share of years read correctly."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Normalize OCR and reference text the same way
    # 2. CER / WER per CV (jiwer); distribution and mean per occupation
    # 3. Share of reference years (19xx/20xx) that appear in the OCR text
    raise NotImplementedError


if __name__ == "__main__":
    main()
