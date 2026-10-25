"""Stage 1 - Run PaddleOCR once per image and cache one JSON per CV in data/ocr/.

Model: PP-OCRv5 mobile detection + English recognition, on CPU. The mobile models were chosen
because OCR runs on a laptop CPU within the deadline; the server models are more accurate but
several times slower without a GPU build of PaddlePaddle.

Each JSON: {id, lines: [{text, score, box: [x0, y0, x1, y1]}], width, height, seconds}.
Usage: python src/ocr_extraction.py [--workers 4] [--limit N]
"""
import argparse
import os
import platform
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from utils import METRICS, OCR_DIR, RAW_IMAGES, ensure_dirs, load_json, save_json

OCR_CONFIG = dict(
    lang="en",
    ocr_version="PP-OCRv5",
    text_detection_model_name="PP-OCRv5_mobile_det",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
_ocr = None


def build_ocr(threads: int = 8):
    from paddleocr import PaddleOCR

    return PaddleOCR(device="cpu", cpu_threads=threads, **OCR_CONFIG)


def ocr_image(ocr, path: str) -> dict:
    """Run OCR on one image and return {lines, width, height, seconds}."""
    t0 = time.perf_counter()
    res = ocr.predict(path)[0]
    secs = time.perf_counter() - t0
    boxes = np.asarray(res["rec_boxes"]).tolist()
    lines = [{"text": t, "score": float(s), "box": [int(v) for v in b]}
             for t, s, b in zip(res["rec_texts"], res["rec_scores"], boxes)]
    w, h = Image.open(path).size
    return {"lines": lines, "width": w, "height": h, "seconds": secs}


def _init(threads: int) -> None:
    global _ocr
    _ocr = build_ocr(threads)


def run_one(path: str) -> dict:
    rid = os.path.splitext(os.path.basename(path))[0]
    out = OCR_DIR / f"{rid}.json"
    if out.exists():
        return {"id": rid, "status": "cached"}
    try:
        save_json({"id": rid, **ocr_image(_ocr, path)}, out)
        return {"id": rid, "status": "ok"}
    except Exception as e:  # keep going, report failures at the end
        return {"id": rid, "status": f"failed: {e}"}


def summarize() -> dict:
    rows = []
    for f in OCR_DIR.glob("*.json"):
        d = load_json(f)
        scores = [ln["score"] for ln in d["lines"]]
        rows.append({"id": d["id"], "seconds": d["seconds"], "n_lines": len(scores),
                     "mean_conf": float(np.mean(scores)) if scores else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "ocr_per_cv.csv", index=False)
    return {
        "n_cvs": len(df),
        "seconds_per_cv_mean": round(df["seconds"].mean(), 2),
        "seconds_per_cv_median": round(df["seconds"].median(), 2),
        "mean_confidence": round(df["mean_conf"].mean(), 4),
        "lines_per_cv_mean": round(df["n_lines"].mean(), 1),
        "empty_outputs": int((df["n_lines"] == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="process only N images (smoke test)")
    args = parser.parse_args()
    ensure_dirs()

    paths = sorted(str(p) for p in RAW_IMAGES.glob("*.png"))[: args.limit]
    threads = max(1, (os.cpu_count() or 4) // args.workers)
    t0 = time.perf_counter()
    results = []
    with Pool(args.workers, initializer=_init, initargs=(threads,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, paths), 1):
            results.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(paths)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    failed = [r for r in results if r["status"].startswith("failed")]
    pd.DataFrame(failed, columns=["id", "status"]).to_csv(METRICS / "ocr_failures.csv", index=False)
    summary = summarize()
    summary.update({
        "failed_images": len(failed),
        "wall_clock_minutes": round((time.perf_counter() - t0) / 60, 1),
        "workers": args.workers, "threads_per_worker": threads,
        "hardware": f"{platform.processor()} ({os.cpu_count()} logical cores), CPU inference",
        "config": OCR_CONFIG,
    })
    save_json(summary, METRICS / "ocr_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
