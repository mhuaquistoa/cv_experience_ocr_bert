"""Shared constants, paths and helpers used by every stage of the pipeline."""
import json
import os
import random
from pathlib import Path

import numpy as np

SEED = 42

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_IMAGES = RAW / "images"
OCR_DIR = DATA / "ocr"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
METRICS = OUTPUT / "metrics"

# Dataset
HF_DATASET = "sumin-yu/PopResume"
OCCUPATIONS = [  # shard prefixes, as listed in data/images/
    "registered_nurses",
    "elementary_middle_school_teachers",
    "software_developers",
    "accountants_auditors",
    "construction_laborers",
]
N_PER_OCCUPATION = 400  # minimum allowed (2,000 total) given the compute/time budget

# Target
TARGET = "exp_year"
CAP = 20  # raise to 25/30 only if >15% of the sample sits at 20, same cap for every model
BANDS = [(0, 2), (3, 5), (6, 10), (11, CAP)]
BAND_LABELS = ["0-2", "3-5", "6-10", f"11-{CAP}"]

# Never use these as model features (sex/race only for the fairness check)
FORBIDDEN_FEATURES = ["age", "sex", "race", "name", "state", "region"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clip_years(y, cap: int = CAP):
    return np.clip(y, 0, cap)


def to_band(y) -> np.ndarray:
    """Map years (continuous or integer) to band indices 0..len(BANDS)-1."""
    y = np.round(clip_years(np.asarray(y, dtype=float)))
    edges = [hi for _, hi in BANDS[:-1]]
    return np.digitize(y, edges, right=True)


def ensure_dirs() -> None:
    for p in [RAW_IMAGES, OCR_DIR, PROCESSED, SPLITS, MODELS / "baselines",
              MODELS / "bert", FIGURES, TABLES, METRICS]:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared data access for the modelling stages
# ---------------------------------------------------------------------------
import re  # noqa: E402

import pandas as pd  # noqa: E402

TEXT_VARIANTS = ["raw", "clean", "clean_conf"]  # Question 3: OCR cleaning sensitivity
MAIN_VARIANT = "clean"
PREDS = METRICS / "preds"
PROBE_SHIFT = 5
N_PROBE = 20


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str})


def shift_years(text: str, shift: int = PROBE_SHIFT) -> str:
    """Probe for Question 1: move every 4-digit year (1950-2029) by `shift` years."""
    return re.sub(r"\b(19[5-9]\d|20[0-2]\d)\b", lambda m: str(int(m.group(1)) + shift), text)


DATE_RANGE = re.compile(r"((?:19|20)\d{2})[.,](\d{1,2})\s*[-–~]\s*((?:19|20)\d{2})[.,](\d{1,2})")
BULLET = re.compile(r"^\s*[-•·*]")


def drop_oldest_job(text: str):
    """Probe for Question 1: delete the job with the earliest start date (its line + bullets).

    Returns (new_text, years_removed); years_removed is None when the CV has fewer than 2 jobs.
    """
    lines = text.split("\n")
    jobs = []
    for i, ln in enumerate(lines):
        m = DATE_RANGE.search(ln)
        if m:
            y0, m0, y1, m1 = map(int, m.groups())
            jobs.append((y0 + m0 / 12, i, (y1 + m1 / 12) - (y0 + m0 / 12)))
    if len(jobs) < 2:
        return text, None
    _, i, dur = min(jobs)
    j = i + 1
    while j < len(lines) and BULLET.match(lines[j]):
        j += 1
    return "\n".join(lines[:i] + lines[j:]), max(dur, 0.0)


EVAL_SETS = ["val", "test", "test_ref", "probe", "probe_drop"]


def load_sets(variant: str = MAIN_VARIANT) -> dict:
    """Return {set_name: DataFrame[id, occupation, exp_year, text]} for every evaluation set.

    Sets: train, val, test, test_ref (test CVs with the reference text, Question 2 only),
    probe (20 test CVs with every year shifted +5) and probe_drop (20 test CVs with >= 2 jobs
    whose oldest job is deleted; column expected_change), both for Question 1 only.
    """
    cvs = read_csv(PROCESSED / "cvs.csv")
    out = {}
    for name in ["train", "val", "test"]:
        ids = read_csv(SPLITS / f"{name}.csv")[["id"]]
        df = ids.merge(cvs, on="id", how="left")
        df["text"] = df[f"text_{variant}"].fillna("")
        out[name] = df
    out["test_ref"] = out["test"].assign(text=out["test"]["text_ref"].fillna(""))
    probe = out["test"].sample(N_PROBE, random_state=SEED)
    out["probe"] = probe.assign(text=probe["text"].map(shift_years))
    dropped = out["test"]["text"].map(drop_oldest_job)
    cand = out["test"].assign(text=[d[0] for d in dropped], expected_change=[-(d[1] or 0) for d in dropped])
    cand = cand[[d[1] is not None for d in dropped]]
    out["probe_drop"] = cand.sample(min(N_PROBE, len(cand)), random_state=SEED)
    return out


def save_predictions(model: str, variant: str, sets: dict, preds: dict, timing: dict) -> None:
    """Save long-format predictions (id, set, y_true, y_pred) and timing for one model/variant."""
    PREDS.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, yhat in preds.items():
        df = sets[name]
        frames.append(pd.DataFrame({"id": df["id"].values, "set": name,
                                    "y_true": df[TARGET].values, "y_pred": clip_years(yhat)}))
    pd.concat(frames).to_csv(PREDS / f"{model}__{variant}.csv", index=False)
    save_json(timing, PREDS / f"{model}__{variant}_timing.json")
