"""Streamlit app - exactly 4 tabs. Run with: streamlit run app.py"""
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from utils import (CAP, FIGURES, METRICS, MODELS, PROCESSED, RAW_IMAGES, TABLES, TARGET,  # noqa: E402
                   load_json, read_csv)

st.set_page_config(page_title="CV Experience Estimator", layout="wide")
st.title("Predicting Years of Experience from CV Images")
st.caption("PaddleOCR + fine-tuned BERT vs median, TF-IDF and sentence-embedding baselines · PopResume (synthetic CVs)")

MODEL_LABELS = {"median": "Median", "tfidf": "TF-IDF + Ridge", "embeddings": "Embeddings + Ridge",
                "bert": "Fine-tuned BERT"}


def jload(path: Path):
    return load_json(path) if path.exists() else None


def table(name: str, **kw):
    p = TABLES / name
    if p.exists():
        return pd.read_csv(p, **kw)
    st.warning(f"`output/tables/{name}` not found - run `python src/evaluation.py`.")
    return None


def figure(name: str, caption: str):
    p = FIGURES / name
    if p.exists():
        st.image(str(p), use_container_width=True)
        st.caption(caption)
    else:
        st.warning(f"`output/figures/{name}` not found - run `python src/visualization.py`.")


def readme_section(title: str) -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else "_Section not found in README._"


tab1, tab2, tab3, tab4 = st.tabs([
    "Problem & Methodology",
    "OCR & Exploratory Analysis",
    "Model Results",
    "Interactive Estimator",
])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.markdown(readme_section("Problem"))
    st.markdown(readme_section("Methodology"))
    st.markdown(readme_section("Limitations"))

# ---------------------------------------------------------------- Tab 2
with tab2:
    cvs_path = PROCESSED / "cvs.csv"
    if cvs_path.exists():
        cvs = read_csv(cvs_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("CVs", len(cvs))
        c2.metric(f"At the cap ({CAP} years)", f"{(cvs[TARGET] >= CAP).sum()} ({(cvs[TARGET] >= CAP).mean():.1%})")
        c3.metric("Median years", f"{cvs[TARGET].median():.0f}")
    figure("experience_distribution.png",
           "A histogram shows the full shape of the target (and the pile-up at the cap); the box plot "
           "compares occupations, which differ in typical career length.")

    q = jload(METRICS / "ocr_quality_summary.json")
    o = jload(METRICS / "ocr_summary.json")
    if q and o:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean CER", f"{q['cer_mean']:.1%}")
        c2.metric("Mean WER", f"{q['wer_mean']:.1%}")
        c3.metric("Years read correctly", f"{q['years_read_correctly']:.1%}")
        c4.metric("OCR seconds per CV", f"{o['seconds_per_cv_mean']:.1f}")
        st.dataframe(pd.DataFrame(q["by_occupation"]).T.style.format("{:.2%}"))
    figure("ocr_quality.png",
           "Error-rate histograms show whether OCR errors are spread thin or concentrated in a few bad "
           "CVs; the year bar chart focuses on the tokens the models actually need.")

    st.subheader("Example: CV image with PaddleOCR boxes")
    ocr_files = sorted((ROOT / "data" / "ocr").glob("*.json"))
    if ocr_files:
        pick = st.selectbox("CV id", [f.stem for f in ocr_files[:200]])
        doc = load_json(ROOT / "data" / "ocr" / f"{pick}.json")
        from preprocessing import prepare, reading_order

        col_img, col_txt = st.columns(2)
        img_path = RAW_IMAGES / f"{pick}.png"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for ln in doc["lines"]:
                draw.rectangle(ln["box"], outline=(255, 0, 0) if ln["score"] < 0.8 else (0, 160, 0), width=2)
            col_img.image(img, caption="green: confidence >= 0.8, red: < 0.8", use_container_width=True)
        col_txt.text_area("Reading order (raw OCR)", reading_order(doc), height=400)
        col_txt.text_area("Model input (cleaned + masked)", prepare(doc), height=300)
    else:
        st.info("No OCR output yet.")

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Model comparison (test set)")
    comp = table("model_comparison.csv")
    if comp is not None:
        st.dataframe(comp.set_index("Model").style.format("{:.3f}"))
    st.subheader("BERT's MAE gain over each baseline (95% paired bootstrap CI)")
    gain = table("bert_gain_bootstrap.csv")
    if gain is not None:
        st.dataframe(gain)

    st.subheader("Predicted vs true")
    available = [k for k in MODEL_LABELS if (METRICS / "preds" / f"{k}__clean.csv").exists()]
    if available:
        k = st.selectbox("Model", available, format_func=MODEL_LABELS.get)
        d = read_csv(METRICS / "preds" / f"{k}__clean.csv").query("set == 'test'")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(d["y_true"], d["y_pred"], s=12, alpha=0.5)
        ax.plot([0, CAP], [0, CAP], "k--", lw=1)
        ax.set(xlabel="true years", ylabel="predicted years", xlim=(-1, CAP + 1), ylim=(-1, CAP + 1))
        st.pyplot(fig, use_container_width=False)
        st.caption("Points above the dashed line are overestimates. A flat cloud means the model pulls "
                   "every CV toward the middle (overestimating juniors, underestimating seniors).")
        figure(f"confusion_{k}.png", "Band confusion matrix: off-diagonal cells are band errors.")

    c1, c2 = st.columns(2)
    with c1:
        figure("mae_by_band.png", "Where each model's errors come from along the career.")
    with c2:
        figure("mae_by_occupation.png", "Whether some occupations are harder to read.")

    st.subheader("Sensitivity (Question 3): test MAE per setup")
    sens = table("sensitivity.csv", index_col=0)
    if sens is not None:
        st.dataframe(sens)
    st.subheader("Fairness check (Question 4): BERT by sex and race")
    fair = table("fairness.csv")
    if fair is not None:
        st.dataframe(fair)
    st.subheader("Analytical questions")
    st.markdown(readme_section("Analytical questions"))

# ---------------------------------------------------------------- Tab 4


@st.cache_resource
def load_models():
    import joblib

    models = {}
    med = jload(MODELS / "baselines" / "median.json")
    if med:
        models["median"] = lambda texts: np.full(len(texts), med["median"])
    if (MODELS / "baselines" / "tfidf.joblib").exists():
        tfidf = joblib.load(MODELS / "baselines" / "tfidf.joblib")
        models["tfidf"] = tfidf.predict
    if (MODELS / "baselines" / "embeddings_ridge.joblib").exists():
        from baseline_embeddings import embed, load_encoder

        enc, ridge = load_encoder(), joblib.load(MODELS / "baselines" / "embeddings_ridge.joblib")
        models["embeddings"] = lambda texts: ridge.predict(embed(enc, texts))
    if (MODELS / "bert" / "best").exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from train_bert import predict

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODELS / "bert" / "best")
        bert = AutoModelForSequenceClassification.from_pretrained(MODELS / "bert" / "best").to(dev)
        models["bert"] = lambda texts: predict(bert, tok, texts, "head_tail", dev)
    return models


@st.cache_resource
def load_ocr():
    from ocr_extraction import build_ocr

    return build_ocr()


with tab4:
    st.subheader("Estimate experience from a CV")
    up = st.file_uploader("CV image (PNG/JPG) or PDF (first page)", type=["png", "jpg", "jpeg", "pdf"])
    if up is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.png"
            if up.name.lower().endswith(".pdf"):
                import fitz  # pymupdf

                page = fitz.open(stream=up.read(), filetype="pdf")[0]
                page.get_pixmap(dpi=150).save(str(path))
            else:
                Image.open(up).convert("RGB").save(path)
            from ocr_extraction import ocr_image
            from preprocessing import prepare

            with st.spinner("Running PaddleOCR..."):
                doc = ocr_image(load_ocr(), str(path))
            text = prepare(doc)
            c1, c2 = st.columns(2)
            c1.image(str(path), use_container_width=True)
            c2.text_area("Extracted text (cleaned + masked)", text, height=500)
        models = load_models()
        if models:
            cols = st.columns(len(models))
            for col, (k, f) in zip(cols, models.items()):
                col.metric(MODEL_LABELS[k], f"{float(np.clip(f([text])[0], 0, CAP)):.1f} years")
        else:
            st.warning("No trained models found in models/.")

    st.subheader("Browse test CVs")
    frames = {k: read_csv(METRICS / "preds" / f"{k}__clean.csv").query("set == 'test'").set_index("id")["y_pred"]
              for k in MODEL_LABELS if (METRICS / "preds" / f"{k}__clean.csv").exists()}
    if "bert" in frames and (PROCESSED / "cvs.csv").exists():
        cvs = read_csv(PROCESSED / "cvs.csv").set_index("id")
        browse = pd.DataFrame(frames).rename(columns=MODEL_LABELS).join(cvs[["occupation", TARGET, "text_clean"]])
        browse["BERT |error|"] = (browse["Fine-tuned BERT"] - browse[TARGET]).abs()
        f1, f2, f3 = st.columns(3)
        occ = f1.multiselect("Occupation", sorted(browse["occupation"].unique()))
        yrs = f2.slider("True years", 0, CAP, (0, CAP))
        err = f3.slider("BERT |error| at least", 0.0, float(CAP), 0.0, 0.5)
        sel = browse[browse[TARGET].between(*yrs) & (browse["BERT |error|"] >= err)]
        if occ:
            sel = sel[sel["occupation"].isin(occ)]
        st.dataframe(sel.drop(columns="text_clean").round(1), height=300)
        if len(sel):
            pick = st.selectbox("Inspect CV", sel.index)
            st.text_area("Model input", sel.loc[pick, "text_clean"], height=250)

        st.subheader("Good and bad predictions")
        top = table("bert_top10_errors.csv")
        best = browse.nsmallest(3, "BERT |error|")
        st.markdown("**Best BERT predictions** - dates are read cleanly and the career is continuous:")
        st.dataframe(best.drop(columns="text_clean").round(1))
        if top is not None:
            st.markdown("**Largest BERT errors** with their diagnosed cause:")
            st.dataframe(top.drop(columns=["text_excerpt"], errors="ignore"))
    else:
        st.info("Run the pipeline (BERT predictions) to browse test CVs.")
