# cv_experience_ocr_bert

Predicting a candidate's years of work experience (0–20) from a **CV image**: **PaddleOCR** reads
the document and a **fine-tuned DistilBERT regressor** is compared against a median predictor,
TF-IDF + Ridge and sentence embeddings + Ridge, all trained and tested on the same split.
Data: PopResume (Yu, Park & Moon, EMNLP 2026, CC BY 4.0) — synthetic resumes, no real person's data.

## Problem

**What it does and for whom.** Recruiting agencies, job boards, public employment services and HR
teams receive thousands of CVs, often as scans or images. An experience estimator can **pre-fill a
candidate profile** or **sort a queue for a human reviewer**. It must never decide who gets an interview.

**Why it is hard.** The number of years is rarely written on the CV. A recruiter infers it by
connecting scattered evidence: the dates of each job, how many positions there are, how job titles
progress, and when education ended. Gaps, overlapping jobs and "2019 – Present" make simple
keyword matching insufficient. The question of this project is whether a transformer actually
learns to make these connections better than simpler models, and what OCR noise costs.

## Methodology

**Data and sampling.** 400 CVs per occupation (registered nurses, elementary/middle school
teachers, software developers, accountants & auditors, construction laborers) = **2,000 CVs**, the
minimum allowed: OCR runs on a laptop CPU and the deadline left no time for 4,000. Only images
**without a photo** (`has_photo == False`), streamed per occupation, shuffled with **seed 42**;
the sampled IDs are saved in `data/raw/sampled_ids.csv`.

**Target.** `exp_year`, clipped to 0–20 ("20 or more"); predictions are clipped to the same range.
CVs at the cap: **TBD** (TBD%). The cap was kept at 20 because TBD.

**Masking rule.** Model input comes only from PaddleOCR. Age, sex, race, names and location are
never features (sex and race are used only for the fairness check). Phrases that state the answer —
total experience ("12 years of experience"), age, date of birth — are replaced by `[MASKED]` with
regular expressions (`src/preprocessing.py`); job dates and graduation years stay because they are
the evidence. CVs affected: **TBD** (total experience TBD, age TBD, date of birth TBD). Fifty random
OCR texts (`data/processed/review_sample.txt`) were read to design these rules.

**OCR.** PaddleOCR 3.x with `lang="en"`, PP-OCRv5 **mobile** detection + English recognition, no
orientation/unwarping (CVs are clean digital renders). Mobile instead of server models because OCR
runs on CPU (20 logical cores, 4 parallel workers); server models are more accurate but several
times slower without a GPU build of Paddle. Every line's text, confidence and box is cached as one
JSON per CV in `data/ocr/`. Time per CV: **TBD s**, mean confidence **TBD**, failed images **TBD**.

**Reading order.** Lines are grouped into visual rows (same y-centre within half a line height)
and read top-to-bottom, left-to-right. **Two-column layouts** are detected with an x-coverage
histogram of the boxes: if an (almost) empty vertical gutter exists in the central 20–80% of the
page with ≥20% of the lines on each side, the left column is read before the right one. Lines that
cross the gutter (full-width section headers) act as separators, so a job title stays next to its dates.

**Cleaning.** De-hyphenation across lines, unified dashes in date ranges, whitespace normalization,
and removal of the name line, emails, URLs, phone numbers, street addresses and "City, ST ZIP".
Low-confidence lines are **kept** in the main setup (dropping them can delete a date); a variant that
drops lines with confidence < 0.80 is evaluated in Question 3.

**Split.** 70/15/15 by CV ID, stratified by occupation × experience band (0–2, 3–5, 6–10, 11–20),
seed 42 (`data/splits/`). All tuning uses the validation set; the test set is used once at the end.

**Models.**
| Model | Setup |
|---|---|
| Median | Always predicts the training-set median (TBD years). |
| TF-IDF + Ridge | Word 1–2-grams + char 3–5-grams, sublinear TF. **Digits and years kept** as tokens (they are the evidence); removing them is tested in Q3. Ridge α tuned on validation. |
| Embeddings + Ridge | Frozen `BAAI/bge-small-en-v1.5`. CVs longer than the 512-token limit are split into 250-word windows; the CV vector is the mean of the window embeddings. Ridge α tuned on validation. |
| Fine-tuned BERT | `distilbert-base-uncased`, `num_labels=1`, `problem_type="regression"`, MSE loss, target ÷ 20 during training. Max length 512 with **head + tail** truncation (first 320 + last 190 tokens): the head holds recent jobs, the tail holds the oldest jobs and education. lr 5e-5, batch 16, up to 10 epochs, early stopping on validation MAE (patience 2), seed 42, fp16 on an RTX 5050 Laptop GPU (8 GB). CVs longer than 512 tokens: TBD%; years surviving truncation: TBD%. Training time: TBD. |

## Results

Test set (n = TBD). Full table: `output/tables/model_comparison.csv`.

| Model | MAE | RMSE | Acc ±1 | Acc ±2 | Acc ±3 | Band acc. | Band macro-F1 | R² | Train time | Infer./CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Median | TBD | | | | | | | | | |
| TF-IDF + Ridge | TBD | | | | | | | | | |
| Embeddings + Ridge | TBD | | | | | | | | | |
| Fine-tuned BERT | TBD | | | | | | | | | |

**BERT's MAE gain** (95% paired bootstrap CI, 2,000 resamples): vs median TBD [TBD, TBD];
vs TF-IDF TBD [TBD, TBD]; vs embeddings TBD [TBD, TBD].

**OCR quality.** Mean CER **TBD**, mean WER **TBD**, years read correctly **TBD**
(per occupation in `output/metrics/ocr_quality_summary.json`).

**Error analysis.** The 10 largest BERT errors (`output/tables/bert_top10_errors.csv`) grouped by
cause: TBD.

Figures: `output/figures/` (predicted vs true, MAE by band and occupation, confusion matrices).

## Analytical questions

**Q1 — What does BERT learn that the baselines cannot?** TBD: MAE reduction vs each baseline and
significance; MAE by band (juniors / mid-career / seniors); probe (every year +5 on 20 test CVs,
`output/tables/probe_shift_years.csv`): a recruiter would lower the estimate for CVs with an
open-ended "Present" job and leave the rest unchanged — TBD whether each model does.

**Q2 — How much does OCR noise cost?** TBD: CER / WER / years read correctly and where OCR fails;
MAE on OCR text vs reference text for each model (`output/tables/ocr_vs_reference.csv`); most robust model.

**Q3 — Methodological sensitivity.** OCR cleaning (raw vs cleaned vs low-confidence lines removed)
for every model, plus digits in TF-IDF, MSE vs Huber and head vs head + tail for BERT
(`output/tables/sensitivity.csv`). TBD.

**Q4 — Business value, fairness and ethics.** Best use: pre-filling a profile or ordering a
reviewer's queue, with a human always checking. An error of ±2 years rarely changes a seniority
level; ±5 can move a candidate between junior and senior. **Underestimating is worse** in screening:
it silently filters out qualified people, while an overestimate is caught at the interview.
Fairness (`output/tables/fairness.csv`): BERT MAE and mean error by sex and race — TBD. Sex and race
are not inputs, but they can leak through proxies (names are removed, but occupations, schools,
career gaps linked to caregiving, or OCR quality of certain layouts may correlate with groups).

## Limitations

Synthetic resumes generated from U.S. statistics (real CVs are messier); only 5 occupations;
English only; 2,000 CVs instead of 4,000; OCR on CPU with mobile models; a single seed per model.
Such a system must **never decide who is hired**: it estimates one number from a document, cannot
see context (part-time work, career breaks, unpaid work) and may carry group-level bias.

**Possible improvements:** server OCR models on GPU, a layout-aware model (LayoutLM), a rule-based
date-arithmetic feature fed to BERT, longer-context models (ModernBERT), and more data.

## Reproduce

```bash
conda create -n cvexp python=3.11 -y && conda activate cvexp
# PaddlePaddle first (CPU build), then the rest:
pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; omit the index for CPU
pip install -r requirements.txt
huggingface-cli login            # after accepting the PopResume access conditions

python src/download_data.py      # 400 no-photo CVs per occupation -> data/raw/
python src/ocr_extraction.py     # PaddleOCR -> data/ocr/ (cached, re-runnable)
python src/preprocessing.py      # reading order, cleaning, masking -> data/processed/
python src/ocr_quality.py        # CER / WER / years read correctly
python src/split.py              # 70/15/15 -> data/splits/
python src/baseline_median.py
python src/baseline_tfidf.py
python src/baseline_tfidf.py --no-digits            # Q3
python src/baseline_embeddings.py
python src/train_bert.py                            # main model
python src/train_bert.py --variant raw              # Q3
python src/train_bert.py --variant clean_conf       # Q3
python src/train_bert.py --loss huber               # Q3
python src/evaluation.py         # -> output/tables/
python src/visualization.py      # -> output/figures/
```

Images (`data/raw/`) and model weights (`models/`) are not committed; the commands above regenerate them.

## Run the app

```bash
streamlit run app.py
```

Four tabs: Problem & Methodology · OCR & Exploratory Analysis · Model Results · Interactive Estimator
(upload a CV image or PDF, PaddleOCR runs, and all four models predict side by side).

## Workflow

No direct commits to `main`: each stage is developed on a `feature/...` branch and merged through a Pull Request.
