# cv_experience_ocr_bert

Predicting years of work experience (0–20) from CV images: **PaddleOCR** extracts the text and a
**fine-tuned BERT regressor** is compared against a median predictor, TF-IDF + Ridge and
sentence embeddings + Ridge. Data: PopResume (Yu, Park & Moon, EMNLP 2026, CC BY 4.0).

> Work in progress. Sections below are filled in as each stage is completed.

## Setup

```bash
conda create -n cvexp python=3.11 -y
conda activate cvexp
# PaddlePaddle first (see the comments in requirements.txt for CPU/GPU builds)
pip install -r requirements.txt
huggingface-cli login   # accept the PopResume access conditions first
```

## Reproduce

```bash
python src/download_data.py        # stratified sample -> data/raw/
python src/ocr_extraction.py       # PaddleOCR -> data/ocr/ (cached)
python src/ocr_quality.py          # CER / WER / years read correctly
python src/preprocessing.py        # reading order, cleaning, masking -> data/processed/
python src/split.py                # 70/15/15 -> data/splits/
python src/baseline_median.py
python src/baseline_tfidf.py
python src/baseline_embeddings.py
python src/train_bert.py
python src/evaluation.py           # -> output/tables/model_comparison.csv
python src/visualization.py        # -> output/figures/
streamlit run app.py
```

Images (`data/raw/`) and model weights (`models/`) are not committed; the commands above regenerate them.

## Contents (to complete)

1. What the project does and for whom
2. Sampling and experience distribution
3. PaddleOCR model, settings and reading order
4. OCR quality (CER, WER, years read correctly)
5. Cleaning and masking (number of CVs masked)
6. Split
7. Baselines
8. BERT setup and long-CV strategy
9. Final metrics and BERT's MAE gain (95% bootstrap CI)
10. Analytical questions 1–4
11. Limitations and business applications

## Workflow

No direct commits to `main`: each stage is developed on a `feature/...` branch and merged through a Pull Request.
