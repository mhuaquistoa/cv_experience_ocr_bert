"""Stage 6 - Evaluation of every model on the same test set + analyses for Questions 1-4.

Output (output/tables/):
    model_comparison.csv      main table (MAE, RMSE, Acc +-k, band acc / macro-F1, R2, times)
    bert_gain_bootstrap.csv   MAE(baseline) - MAE(BERT), 95% paired bootstrap CI (2,000 resamples)
    mae_by_band.csv, mae_by_occupation.csv, confusion_<model>.csv
    bert_top10_errors.csv     for manual error analysis (fill the 'cause' column)
    probe_shift_years.csv     Question 1: prediction change when every year moves +5
    ocr_vs_reference.csv      Question 2: MAE on OCR text vs reference text
    sensitivity.csv           Question 3: test MAE per model x setup
    fairness.csv              Question 4: BERT MAE and mean error by sex and race
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score

from utils import (BAND_LABELS, MAIN_VARIANT, PREDS, PROCESSED, SEED, TABLES, TARGET, drop_oldest_job, ensure_dirs,
                   load_json, read_csv, to_band)

MAIN_MODELS = {"median": "Median", "tfidf": "TF-IDF + Ridge", "embeddings": "Embeddings + Ridge",
               "bert": "Fine-tuned BERT", "rules": "Rule-based date sum (bonus)"}
N_BOOT = 2000


def load_preds(model: str, variant: str = MAIN_VARIANT):
    path = PREDS / f"{model}__{variant}.csv"
    if not path.exists():
        return None, None
    return read_csv(path), load_json(PREDS / f"{model}__{variant}_timing.json")


def metrics(y, p) -> dict:
    err = np.abs(p - y)
    bt, bp = to_band(y), to_band(p)
    return {
        "MAE (years)": err.mean(),
        "RMSE (years)": np.sqrt(((p - y) ** 2).mean()),
        "Acc ±1": (err <= 1).mean(), "Acc ±2": (err <= 2).mean(), "Acc ±3": (err <= 3).mean(),
        "Band accuracy": accuracy_score(bt, bp),
        "Band macro-F1": f1_score(bt, bp, average="macro", labels=range(len(BAND_LABELS)), zero_division=0),
        "R²": r2_score(y, p),
    }


def bootstrap_gain(y, p_base, p_bert, rng) -> tuple:
    e_base, e_bert = np.abs(p_base - y), np.abs(p_bert - y)
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    diffs = e_base[idx].mean(axis=1) - e_bert[idx].mean(axis=1)
    return e_base.mean() - e_bert.mean(), *np.percentile(diffs, [2.5, 97.5])


def main() -> None:
    ensure_dirs()
    cvs = read_csv(PROCESSED / "cvs.csv").set_index("id")
    rng = np.random.default_rng(SEED)

    test = {}
    rows = []
    for key, label in MAIN_MODELS.items():
        df, timing = load_preds(key)
        if df is None:
            print(f"(missing predictions for {key})")
            continue
        t = df[df["set"] == "test"].set_index("id")
        test[key] = t
        rows.append({"Model": label, **metrics(t["y_true"].values, t["y_pred"].values),
                     "Training time (s)": timing["train_seconds"],
                     "Inference time per CV (s)": timing["infer_seconds_per_cv"]})
    comp = pd.DataFrame(rows)
    comp.round(4).to_csv(TABLES / "model_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))

    ids = test["median"].index
    y = test["median"].loc[ids, "y_true"].values

    # BERT gain with paired bootstrap CI
    if "bert" in test:
        pb = test["bert"].loc[ids, "y_pred"].values
        gains = []
        for key in ["median", "tfidf", "embeddings", "rules"]:
            if key in test:
                g, lo, hi = bootstrap_gain(y, test[key].loc[ids, "y_pred"].values, pb, rng)
                gains.append({"Baseline": MAIN_MODELS[key], "MAE reduction (years)": g, "CI 2.5%": lo,
                              "CI 97.5%": hi, "Significant (CI excludes 0)": bool(lo > 0 or hi < 0)})
        pd.DataFrame(gains).round(3).to_csv(TABLES / "bert_gain_bootstrap.csv", index=False)
        print(pd.DataFrame(gains).round(3).to_string(index=False))

    # MAE by band / occupation, confusion matrices
    meta = cvs.loc[ids]
    band = pd.Series(np.array(BAND_LABELS)[to_band(y)], index=ids)
    by_band, by_occ = {}, {}
    for key, t in test.items():
        err = (t.loc[ids, "y_pred"] - t.loc[ids, "y_true"]).abs()
        by_band[MAIN_MODELS[key]] = err.groupby(band).mean()
        by_occ[MAIN_MODELS[key]] = err.groupby(meta["occupation"]).mean()
        cm = confusion_matrix(to_band(y), to_band(t.loc[ids, "y_pred"].values), labels=range(len(BAND_LABELS)))
        pd.DataFrame(cm, index=BAND_LABELS, columns=BAND_LABELS).to_csv(TABLES / f"confusion_{key}.csv")
    pd.DataFrame(by_band).reindex(BAND_LABELS).round(3).to_csv(TABLES / "mae_by_band.csv")
    pd.DataFrame(by_occ).round(3).to_csv(TABLES / "mae_by_occupation.csv")

    # Top-10 BERT errors and fairness
    if "bert" in test:
        b = test["bert"].loc[ids].copy()
        b["error"] = b["y_pred"] - b["y_true"]
        top = b.reindex(b["error"].abs().sort_values(ascending=False).index).head(10)
        top = top.join(cvs[["occupation", "text_clean"]])
        top["text_excerpt"] = top["text_clean"].str.slice(0, 600)
        top["cause"] = ""  # fill manually: OCR date / reading order / gap / overlap / missing dates / truncation / other
        top.drop(columns=["text_clean", "set"]).round(2).to_csv(TABLES / "bert_top10_errors.csv")

        fair = []
        for attr in ["sex", "race"]:
            for grp, sub in b.join(cvs[[attr]]).groupby(attr):
                fair.append({"attribute": attr, "group": grp, "n": len(sub),
                             "MAE": sub["error"].abs().mean(), "mean error (pred - true)": sub["error"].mean()})
        pd.DataFrame(fair).round(3).to_csv(TABLES / "fairness.csv", index=False)

    # Question 1 probes, Question 2 OCR vs reference
    expected_drop = {rid: -(drop_oldest_job(t)[1] or 0) for rid, t in cvs["text_clean"].items()}
    probe_rows, ocr_rows = [], []
    for key, label in MAIN_MODELS.items():
        df, _ = load_preds(key)
        if df is None:
            continue
        base = df[df["set"] == "test"].set_index("id")["y_pred"]
        # (a) every year +5: durations are unchanged, so a recruiter's estimate would not move
        pr = df[df["set"] == "probe"].set_index("id")["y_pred"]
        shift = pr - base.loc[pr.index]
        # (b) oldest job deleted: a recruiter would lower the estimate by about that job's length
        pd_ = df[df["set"] == "probe_drop"].set_index("id")["y_pred"]
        drop = pd_ - base.loc[pd_.index]
        exp = pd.Series({i: expected_drop[i] for i in pd_.index})
        probe_rows.append({"Model": label,
                           "+5 shift: mean |change| (ideal 0)": shift.abs().mean(),
                           "drop job: mean change": drop.mean(),
                           "drop job: expected change": exp.mean(),
                           "drop job: share moving down": (drop < -0.1).mean(),
                           "drop job: corr(change, expected)": drop.corr(exp) if drop.std() > 0 else np.nan,
                           "n per probe": len(pr)})
        ref = df[df["set"] == "test_ref"]
        mae_ocr = (base.loc[ids] - test[key].loc[ids, "y_true"]).abs().mean()
        mae_ref = (ref["y_pred"] - ref["y_true"]).abs().mean()
        ocr_rows.append({"Model": label, "MAE on OCR text": mae_ocr, "MAE on reference text": mae_ref,
                         "Gap (OCR - reference)": mae_ocr - mae_ref})
    pd.DataFrame(probe_rows).round(3).to_csv(TABLES / "probe_shift_years.csv", index=False)
    pd.DataFrame(ocr_rows).round(3).to_csv(TABLES / "ocr_vs_reference.csv", index=False)

    # Question 3 sensitivity: every predictions file available
    sens = []
    for f in sorted(PREDS.glob("*__*.csv")):
        model, variant = f.stem.split("__")
        t = read_csv(f).query("set == 'test'")
        sens.append({"model": model, "setup": variant, "test MAE": (t["y_pred"] - t["y_true"]).abs().mean()})
    sens = pd.DataFrame(sens)
    sens.pivot(index="model", columns="setup", values="test MAE").round(3).to_csv(TABLES / "sensitivity.csv")
    print(sens.pivot(index="model", columns="setup", values="test MAE").round(3))


if __name__ == "__main__":
    main()
