"""Stage 2b - Reading order, cleaning, PII removal and masking of direct answers.

Three text variants are produced for the OCR-cleaning sensitivity analysis (Question 3):
    raw         reading order only (+ mandatory masking)
    clean       + de-hyphenation, name/PII removal, whitespace normalization   <- main variant
    clean_conf  clean, after dropping lines with confidence < CONF_THRESHOLD
The reference text gets the same cleaning (column text_ref) for the OCR-noise analysis (Question 2).

Output: data/processed/cvs.csv, data/processed/review_sample.txt (50 random CVs to read),
        output/metrics/masking_report.json
"""
import re

import numpy as np
import pandas as pd

from utils import (METRICS, OCR_DIR, PROCESSED, RAW, SEED, TARGET, ensure_dirs, load_json,
                   read_csv, save_json)

CONF_THRESHOLD = 0.80

# --------------------------------------------------------------------------- reading order


def _rows(lines: list) -> list:
    """Group lines into visual rows (y-centre within half a median line height), left to right."""
    if not lines:
        return []
    med_h = np.median([ln["box"][3] - ln["box"][1] for ln in lines]) or 10
    lines = sorted(lines, key=lambda ln: (ln["box"][1] + ln["box"][3]) / 2)
    rows, cur = [], [lines[0]]
    cur_y = (lines[0]["box"][1] + lines[0]["box"][3]) / 2
    for ln in lines[1:]:
        y = (ln["box"][1] + ln["box"][3]) / 2
        if abs(y - cur_y) <= med_h / 2:
            cur.append(ln)
        else:
            rows.append(sorted(cur, key=lambda l: l["box"][0]))
            cur, cur_y = [ln], y
    rows.append(sorted(cur, key=lambda l: l["box"][0]))
    return rows


def find_column_gap(lines: list, width: int):
    """Return the x of a vertical gutter separating two columns, or None.

    Builds an x-coverage histogram of the line boxes and looks for the widest (almost) empty band
    in the central 20-80% of the page; both sides must hold at least 20% of the lines.
    """
    if len(lines) < 10 or not width:
        return None
    cover = np.zeros(width + 1, dtype=int)
    for ln in lines:
        x0, _, x1, _ = ln["box"]
        cover[max(0, x0):min(width, x1) + 1] += 1
    lo, hi = int(0.2 * width), int(0.8 * width)
    tol = max(3, int(0.15 * len(lines)))  # name, headers and summary lines may cross the gutter
    empty = np.append(cover[lo:hi] <= tol, False)
    best_len, best_start, start = 0, None, None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start > best_len:
                best_len, best_start = i - start, start
            start = None
    if best_len < 0.02 * width:
        return None
    gap = lo + best_start + best_len // 2
    left = sum(ln["box"][2] <= gap for ln in lines)
    right = sum(ln["box"][0] >= gap for ln in lines)
    if min(left, right) < 0.2 * len(lines):
        return None
    return gap


def reading_order(doc: dict, min_conf: float = 0.0) -> str:
    """Top-to-bottom, left-to-right; two-column pages are read column by column.

    Lines crossing the gutter (full-width headers) act as separators: between two separators the
    left column is read fully before the right one, so a job title stays next to its dates.
    """
    lines = [ln for ln in doc["lines"] if ln["score"] >= min_conf and ln["text"].strip()]
    gap = find_column_gap(lines, doc.get("width", 0))
    if gap is None:
        return "\n".join(" ".join(l["text"] for l in row) for row in _rows(lines))

    out, block = [], []

    def flush():
        left = [l for l in block if l["box"][2] <= gap]
        right = [l for l in block if l["box"][2] > gap]
        for part in (left, right):
            out.extend(" ".join(l["text"] for l in row) for row in _rows(part))

    for ln in sorted(lines, key=lambda l: l["box"][1]):
        if ln["box"][0] < gap < ln["box"][2]:  # spans both columns
            flush()
            block = []
            out.append(ln["text"])
        else:
            block.append(ln)
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------- cleaning

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S*", re.I)
PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
STREET = re.compile(r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|"
                    r"Drive|Ln|Lane|Ct|Court|Way|Pl|Place)\b\.?(?:,?\s*(?:Apt|Suite|Unit)\s*\w+)?")
STATE_USA = re.compile(r"\b[A-Z]{2},?\s*USA\b")  # PopResume header line: "AL, USA | phone | email"
CITY_STATE_ZIP = re.compile(r"\b[A-Z][a-zA-Z .]+,\s*[A-Z]{2}(?:\s*\d{5}(?:-\d{4})?)?\b(?!\w)")

# Direct answers: total experience, age, date of birth. Job dates and graduation years stay.
MASK_PATTERNS = {
    "total_experience": re.compile(
        r"\b(?:over|more than|nearly|almost|about|around)?\s*\d{1,2}\+?\s*(?:years?|yrs?)\.?"
        r"(?:\s+(?:of|in|as|working))?(?:\s+[\w/&-]+){0,4}?\s+experience\b"
        r"|\bexperience\s+(?:of\s+)?(?:over\s+|more than\s+)?\d{1,2}\+?\s*(?:years?|yrs?)\b", re.I),
    "age": re.compile(r"\bage[:\s]+\d{2}\b|\b\d{2}\s*(?:years?|yrs?)[\s-]*old\b", re.I),
    "date_of_birth": re.compile(
        r"\b(?:date of birth|d\.o\.b\.?|dob|born(?: on| in)?)\s*[:\-]?\s*[\w ,./-]{0,20}?\d{4}\b", re.I),
}
MASK = "[MASKED]"


def mask_direct_answers(text: str):
    hits = {}
    for name, pat in MASK_PATTERNS.items():
        text, n = pat.subn(MASK, text)
        hits[name] = n
    return text, hits


def remove_pii(text: str) -> str:
    for pat in (EMAIL, URL, PHONE, STATE_USA, STREET, CITY_STATE_ZIP):
        text = pat.sub(" ", text)
    return text


def drop_name_line(text: str) -> str:
    """The candidate's name is one of the first lines in PopResume layouts (2-4 capitalized words)."""
    lines = text.split("\n")
    for i, ln in enumerate(lines[:3]):
        words = ln.split()
        if 1 < len(words) <= 4 and not re.search(r"\d", ln) and all(w[:1].isupper() for w in words):
            return "\n".join(lines[:i] + lines[i + 1:])
    return text


GLUED_RANGE = re.compile(r"\b((?:19|20)\d{2}\.\d{2})\s?((?:19|20)\d{2}\.\d{2})\b")


def repair_date_ranges(text: str) -> str:
    """PaddleOCR drops the en dash in 'YYYY.MM–YYYY.MM' (read as 'YYYY.MMYYYY.MM'); put it back."""
    return GLUED_RANGE.sub(r"\1-\2", text)


def normalize(text: str) -> str:
    text = repair_date_ranges(text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # hyphenation across lines
    text = re.sub(r"[‒-―−]", "-", text)    # unify dashes in date ranges
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean(text: str) -> str:
    return normalize(remove_pii(drop_name_line(text)))


def prepare(doc: dict, variant: str = "clean") -> str:
    """OCR JSON -> model-ready text for one variant (used by the app on uploaded CVs)."""
    if variant == "raw":
        text = reading_order(doc)
    elif variant == "clean_conf":
        text = clean(reading_order(doc, min_conf=CONF_THRESHOLD))
    else:
        text = clean(reading_order(doc))
    return mask_direct_answers(text)[0]


# --------------------------------------------------------------------------- main


def main() -> None:
    ensure_dirs()
    meta = read_csv(RAW / "metadata.csv")
    ref = read_csv(RAW / "reference_text.csv").set_index("id")["text"]

    rows, mask_log = [], []
    for rid in meta["id"]:
        path = OCR_DIR / f"{rid}.json"
        if not path.exists():
            continue
        doc = load_json(path)
        raw = reading_order(doc)
        variants = {
            "raw": raw,
            "clean": clean(raw),
            "clean_conf": clean(reading_order(doc, min_conf=CONF_THRESHOLD)),
            "ref": clean(str(ref.get(rid, ""))),
        }
        row = {"id": rid}
        for name, txt in variants.items():
            masked, hits = mask_direct_answers(txt)
            row[f"text_{name}"] = masked
            if name == "clean":
                mask_log.append({"id": rid, **hits})
        rows.append(row)

    df = meta.merge(pd.DataFrame(rows), on="id", how="inner")
    df[["id", "occupation", TARGET, "sex", "race", "text_raw", "text_clean", "text_clean_conf",
        "text_ref"]].to_csv(PROCESSED / "cvs.csv", index=False)

    log = pd.DataFrame(mask_log).set_index("id")
    report = {
        "n_cvs": len(df),
        "cvs_with_any_mask": int((log.sum(axis=1) > 0).sum()),
        **{f"cvs_masked_{c}": int((log[c] > 0).sum()) for c in log.columns},
        "conf_threshold": CONF_THRESHOLD,
    }
    save_json(report, METRICS / "masking_report.json")
    print(report)

    sample = df.sample(min(50, len(df)), random_state=SEED)
    with open(PROCESSED / "review_sample.txt", "w", encoding="utf-8") as f:
        for _, r in sample.iterrows():
            f.write(f"===== {r['id']} | {r['occupation']} | {TARGET}={r[TARGET]}\n{r['text_raw']}\n\n")
    print(f"Wrote 50 random OCR texts to {PROCESSED / 'review_sample.txt'} for manual review")


if __name__ == "__main__":
    main()
