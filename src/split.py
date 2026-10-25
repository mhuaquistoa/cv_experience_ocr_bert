"""Stage 3 - 70/15/15 split by CV ID, stratified by occupation x experience band (seed 42).

Output: data/splits/{train,val,test}.csv with id, occupation, exp_year.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import PROCESSED, SEED, SPLITS, TARGET, ensure_dirs, read_csv, to_band


def main() -> None:
    ensure_dirs()
    df = read_csv(PROCESSED / "cvs.csv")[["id", "occupation", TARGET]]
    band = pd.Series(to_band(df[TARGET]), index=df.index).astype(str)
    strata = df["occupation"] + "_" + band
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata.loc[rest.index],
                                 random_state=SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(SPLITS / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} CVs, median {TARGET} = {part[TARGET].median()}")


if __name__ == "__main__":
    main()
