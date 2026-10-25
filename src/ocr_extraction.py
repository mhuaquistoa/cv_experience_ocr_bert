"""Stage 1 - Run PaddleOCR once per image and cache one JSON per CV in data/ocr/.

Each JSON stores, per text line: text, confidence and bounding box, plus OCR time."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False,
    #               use_textline_orientation=False) - document which rec model and why
    # 2. Skip images whose JSON already exists (cache)
    # 3. Log time per CV, hardware, mean confidence and failed images
    raise NotImplementedError


if __name__ == "__main__":
    main()
