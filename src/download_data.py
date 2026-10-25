"""Stage 0 - Stratified sampling of no-photo CVs per occupation from PopResume.

Output:
    data/raw/images/<id>.png
    data/raw/metadata.csv        id, occupation, exp_year (raw and capped), sex, race (fairness only)
    data/raw/sampled_ids.csv     list of sampled IDs (reproducibility)
    data/raw/reference_text.csv  id, text (reference only for OCR quality, never a model feature)

Usage: python src/download_data.py [--n 400]
"""
import argparse
import io
import re
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from utils import CAP, HF_DATASET, N_PER_OCCUPATION, RAW, RAW_IMAGES, SEED, TARGET, ensure_dirs, set_seed

META_COLS = ["sex", "race"]  # kept only for the fairness check


def image_shards() -> dict:
    """Map occupation prefix -> list of hf:// shard paths, discovered from the repo."""
    files = HfApi().list_repo_files(HF_DATASET, repo_type="dataset")
    shards = defaultdict(list)
    for f in files:
        m = re.match(r"data/images/(.+)-\d+\.parquet$", f)
        if m:
            shards[m.group(1)].append(f"hf://datasets/{HF_DATASET}/{f}")
    return dict(shards)


def save_image(img, path) -> None:
    if isinstance(img, dict):  # {"bytes": ..., "path": ...}
        img = Image.open(io.BytesIO(img["bytes"]))
    img.convert("RGB").save(path)


def sample_occupation(occupation: str, files: list, n: int) -> list:
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    ds = ds.filter(lambda r: not r["has_photo"]).shuffle(seed=SEED, buffer_size=5000)
    rows = []
    for r in ds.take(n):
        rid = uid(r["job"], r["id"])
        save_image(r["image"], RAW_IMAGES / f"{rid}.png")
        row = {"id": rid, "occupation": occupation, "exp_year_raw": r[TARGET]}
        row.update({c: r.get(c) for c in META_COLS})
        rows.append(row)
    print(f"  {occupation}: {len(rows)} CVs", flush=True)
    return rows


def uid(job: str, rid) -> str:
    """Resume ids are zero-padded per occupation, so the key is <job>_<id>."""
    return f"{job}_{rid}"


def reference_text(ids: set, occupations: list) -> pd.DataFrame:
    frames = []
    for occ in occupations:
        df = pd.read_parquet(f"hf://datasets/{HF_DATASET}/data/text/{occ}.parquet", columns=["id", "job", "text"])
        df["id"] = [uid(j, i) for j, i in zip(df["job"], df["id"])]
        frames.append(df[df["id"].isin(ids)][["id", "text"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PER_OCCUPATION, help="CVs per occupation")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()

    shards = image_shards()
    print(f"Occupations found: {sorted(shards)}")
    rows = []
    for occupation, files in sorted(shards.items()):
        rows += sample_occupation(occupation, sorted(files), args.n)

    meta = pd.DataFrame(rows)
    meta[TARGET] = meta["exp_year_raw"].clip(0, CAP)
    meta.to_csv(RAW / "metadata.csv", index=False)
    meta[["id", "occupation"]].to_csv(RAW / "sampled_ids.csv", index=False)

    ref = reference_text(set(meta["id"]), sorted(shards))
    ref.to_csv(RAW / "reference_text.csv", index=False)

    n_cap = (meta["exp_year_raw"] >= CAP).sum()
    print(f"\nSampled {len(meta)} CVs, reference text for {len(ref)}")
    print(f"CVs at or above the cap ({CAP}): {n_cap} ({n_cap / len(meta):.1%})")
    print(meta.groupby("occupation")[TARGET].describe().round(1))


if __name__ == "__main__":
    main()
