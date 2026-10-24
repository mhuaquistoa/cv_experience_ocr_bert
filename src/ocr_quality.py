"""Stage 2a - OCR quality: CER / WER vs the reference text and share of years read correctly.

Both texts are normalized the same way (lowercase, unified dashes, collapsed whitespace) before
scoring. The OCR text is taken in rebuilt reading order, so WER also reflects ordering errors.
Output: output/metrics/ocr_quality_per_cv.csv, output/metrics/ocr_quality_summary.json
"""
import re
from collections import Counter

import jiwer
import pandas as pd

from preprocessing import reading_order
from utils import METRICS, OCR_DIR, RAW, ensure_dirs, load_json, read_csv, save_json

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[‒-―−]", "-", text)
    text = re.sub(r"[^\w\s@./+-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def years_recall(ref: str, hyp: str):
    """Share of reference years (with multiplicity) that appear in the OCR text."""
    r, h = Counter(YEAR.findall(ref)), Counter(YEAR.findall(hyp))
    total = sum(r.values())
    return (sum(min(c, h[y]) for y, c in r.items()) / total) if total else float("nan"), total


def main() -> None:
    ensure_dirs()
    meta = read_csv(RAW / "metadata.csv")
    ref = read_csv(RAW / "reference_text.csv").set_index("id")["text"]
    rows = []
    for rid, occ in zip(meta["id"], meta["occupation"]):
        path = OCR_DIR / f"{rid}.json"
        if not path.exists() or rid not in ref.index:
            continue
        r, h = norm(str(ref[rid])), norm(reading_order(load_json(path)))
        yr, n_years = years_recall(r, h)
        rows.append({"id": rid, "occupation": occ, "cer": jiwer.cer(r, h), "wer": jiwer.wer(r, h),
                     "years_recall": yr, "n_ref_years": n_years})
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "ocr_quality_per_cv.csv", index=False)

    by_occ = df.groupby("occupation")[["cer", "wer", "years_recall"]].mean().round(4)
    summary = {
        "n_cvs": len(df),
        "cer_mean": round(df["cer"].mean(), 4), "cer_median": round(df["cer"].median(), 4),
        "wer_mean": round(df["wer"].mean(), 4), "wer_median": round(df["wer"].median(), 4),
        "years_read_correctly": round(df["years_recall"].mean(), 4),
        "cvs_with_all_years_correct": round((df["years_recall"] == 1).mean(), 4),
        "by_occupation": by_occ.to_dict(orient="index"),
    }
    save_json(summary, METRICS / "ocr_quality_summary.json")
    print(pd.Series({k: v for k, v in summary.items() if k != "by_occupation"}))
    print(by_occ)


if __name__ == "__main__":
    main()
