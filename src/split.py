"""Stage 3 - 70/15/15 split by CV ID, stratified by occupation x experience band."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Stratify on occupation + to_band(exp_year), seed = SEED
    # 2. Save data/splits/{train,val,test}.csv with id + exp_year
    raise NotImplementedError


if __name__ == "__main__":
    main()
