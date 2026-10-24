"""Figures for the README, the app and the video. Output: output/figures/*.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from evaluation import MAIN_MODELS, load_preds  # noqa: E402
from utils import CAP, FIGURES, METRICS, PROCESSED, TABLES, TARGET, ensure_dirs, read_csv  # noqa: E402

sns.set_theme(style="whitegrid")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cvs = read_csv(PROCESSED / "cvs.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(cvs[TARGET], bins=range(0, CAP + 2), ax=axes[0])
    axes[0].set(title="Years of experience (all sampled CVs)", xlabel="years (capped at 20)")
    sns.boxplot(data=cvs, y="occupation", x=TARGET, ax=axes[1])
    axes[1].set(title="By occupation", xlabel="years", ylabel="")
    save(fig, "experience_distribution.png")

    q = METRICS / "ocr_quality_per_cv.csv"
    if q.exists():
        oq = read_csv(q)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(oq["cer"], ax=axes[0]).set(title="Character error rate per CV")
        sns.histplot(oq["wer"], ax=axes[1]).set(title="Word error rate per CV")
        sns.barplot(data=oq, y="occupation", x="years_recall", ax=axes[2])
        axes[2].set(title="Share of reference years read correctly", xlim=(0, 1), ylabel="")
        save(fig, "ocr_quality.png")

    preds = {k: load_preds(k)[0] for k in MAIN_MODELS}
    preds = {k: v[v["set"] == "test"] for k, v in preds.items() if v is not None}
    fig, axes = plt.subplots(1, len(preds), figsize=(4.5 * len(preds), 4.5), sharey=True)
    for ax, (k, d) in zip(np.atleast_1d(axes), preds.items()):
        jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(d))
        ax.scatter(d["y_true"] + jitter, d["y_pred"], s=10, alpha=0.5)
        ax.plot([0, CAP], [0, CAP], "k--", lw=1)
        mae = (d["y_pred"] - d["y_true"]).abs().mean()
        ax.set(title=f"{MAIN_MODELS[k]} (MAE {mae:.2f})", xlabel="true years", ylabel="predicted years",
               xlim=(-1, CAP + 1), ylim=(-1, CAP + 1))
    save(fig, "pred_vs_true.png")

    for name, title in [("mae_by_band", "MAE by true experience band"),
                        ("mae_by_occupation", "MAE by occupation")]:
        p = TABLES / f"{name}.csv"
        if p.exists():
            t = pd.read_csv(p, index_col=0)
            ax = t.plot.bar(figsize=(10, 4), rot=0)
            ax.set(title=title, ylabel="MAE (years)", xlabel="")
            save(ax.figure, f"{name}.png")

    for k in preds:
        p = TABLES / f"confusion_{k}.csv"
        if p.exists():
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(pd.read_csv(p, index_col=0), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set(title=MAIN_MODELS[k], xlabel="predicted band", ylabel="true band")
            save(fig, f"confusion_{k}.png")
    print(f"Figures saved to {FIGURES}")


if __name__ == "__main__":
    main()
