"""Stage 6 - MAE, RMSE, Acc +-1/2/3, band accuracy / macro-F1, R2, paired bootstrap CIs.

Output: output/tables/model_comparison.csv and per-band / per-occupation / fairness tables."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Same test set for every model
    # 2. Paired bootstrap (>=1000 resamples) of MAE(baseline) - MAE(BERT)
    # 3. MAE by band, by occupation; BERT MAE + mean error by sex and race
    # 4. 10 largest BERT errors for manual error analysis
    raise NotImplementedError


if __name__ == "__main__":
    main()
