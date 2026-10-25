"""Streamlit app - exactly 4 tabs. Run with: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="CV Experience Estimator", layout="wide")
st.title("Predicting Years of Experience from CV Images")

tab1, tab2, tab3, tab4 = st.tabs([
    "Problem & Methodology",
    "OCR & Exploratory Analysis",
    "Model Results",
    "Interactive Estimator",
])

with tab1:
    st.info("TODO: objective, dataset, target, masking rule, OCR setup, models, split, limitations.")
with tab2:
    st.info("TODO: experience distribution, CER/WER, years read correctly, CV image with OCR boxes.")
with tab3:
    st.info("TODO: comparison table, predicted vs true, MAE by band/occupation, confusion matrices, "
            "BERT gain with CI, sensitivity experiment, fairness check.")
with tab4:
    st.info("TODO: upload CV image/PDF -> PaddleOCR -> predictions from all models; browse test CVs.")
