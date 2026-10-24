"""Stage 0 - Stratified sampling (800 no-photo CVs per occupation) from PopResume.

Output: data/raw/images/<id>.png, data/raw/metadata.csv, data/raw/sampled_ids.csv,
data/raw/reference_text.csv (reference text only for OCR quality, never a feature)."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Stream each occupation's image shards, keep has_photo == False, shuffle(seed=42)
    # 2. take(N_PER_OCCUPATION), save images + metadata (id, occupation, exp_year, sex, race)
    # 3. Load the 'text' config and keep id + text for the sampled CVs
    # 4. Report how many CVs sit at the cap (CAP)
    raise NotImplementedError


if __name__ == "__main__":
    main()
