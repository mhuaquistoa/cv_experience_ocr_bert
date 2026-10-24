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
# Shard prefixes under data/images/<occupation>-NNNN.parquet.
# TODO: verify the exact prefixes on the dataset page (only registered_nurses is confirmed).
OCCUPATIONS = [
    "registered_nurses",
    "elementary_and_middle_school_teachers",
    "software_developers",
    "accountants_and_auditors",
    "construction_laborers",
]
N_PER_OCCUPATION = 800

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
