# cv_experience_ocr_bert

Predicting a candidate's years of work experience (0–20) from a **CV image**: **PaddleOCR** reads
the document and a **fine-tuned DistilBERT regressor** is compared against a median predictor,
TF-IDF + Ridge, sentence embeddings + Ridge and (bonus) a rule-based date calculator, all trained
and tested on the same split. Data: PopResume (Yu, Park & Moon, EMNLP 2026, CC BY 4.0) — synthetic
resumes, no real person's data.

**Headline.** Fine-tuned BERT reaches a test **MAE of 0.31 years** (100% of CVs within ±2 years),
3× better than TF-IDF (1.01) and 4× better than embeddings (1.31); the gain is statistically
significant. But a 20-line rule that adds up the job dates reaches **0.02 years**: on these
synthetic CVs, experience is exactly the sum of the job durations, so simple arithmetic beats BERT.

## Problem

**What it does and for whom.** Recruiting agencies, job boards, public employment services and HR
teams receive thousands of CVs, often as scans or images. An experience estimator can **pre-fill a
candidate profile** or **sort a queue for a human reviewer**. It must never decide who gets an interview.

**Why it is hard.** The number of years is not written on the CV (none of the 2,000 sampled CVs
states it). A recruiter infers it by connecting scattered evidence: the dates of each job, how many
positions there are, gaps between them, and when education ended. The model has to do this
arithmetic from text read by OCR, where a single misread character can break a date.

## Methodology

**Data and sampling.** 400 CVs per occupation (accountants & auditors, construction laborers,
elementary/middle school teachers, registered nurses, software developers) = **2,000 CVs**, the
minimum allowed: OCR runs on a laptop CPU and the deadline left no time for 4,000. Only images
**without a photo** (`has_photo == False`), streamed per occupation and shuffled with **seed 42**
(buffer 5,000); sampled IDs are in `data/raw/sampled_ids.csv`. Resume ids repeat across occupations,
so the key is `<occupation>_<id>`.

**Target.** `exp_year` (continuous, in years), clipped to 0–20; predictions are clipped to the same
range. Mean 6.5, median 5.4 years; occupation means range from 5.8 (construction laborers) to 7.2
(accountants). Only **19 CVs (0.9%) sit at the cap**, far below 15%, so the cap stays at 20.
The sample is skewed toward junior and mid-career profiles (`output/figures/experience_distribution.png`).

**Masking rule.** Model input comes only from PaddleOCR. Age, sex, race, names and location are
never features (sex and race are used only for the fairness check). Phrases that state the answer —
total experience ("12 years of experience"), age, date of birth — are replaced by `[MASKED]` with
regular expressions (`src/preprocessing.py`); job dates and graduation years stay because they are
the evidence. After reading 50 random OCR texts (`data/processed/review_sample.txt`) and searching
1,200 reference texts, **no CV contains such a phrase: 0 of 2,000 CVs needed masking**. The rules
stay in the pipeline (and in the app) for real CVs.

**OCR.** PaddleOCR 3.7 (PaddlePaddle 3.1, CPU) with `PP-OCRv5_mobile_det` +
`en_PP-OCRv5_mobile_rec`, oneDNN on, no orientation/unwarping/textline-orientation (CVs are clean
digital renders). The library's default recognizer (`PP-OCRv6_medium_rec`) took ~12 s per CV on
this laptop (45 s with 4 parallel workers); the English mobile model takes ~1.4 s and reads these
documents almost perfectly, so it was chosen. Hardware: Intel Core (20 logical cores), 4 workers ×
5 threads; **4.1 s per CV per worker (~1 s effective), ~35 min for 2,000 CVs, mean line confidence
0.991, 14.4 lines per CV, 0 failed images**. Every line's text, confidence and box is cached as one
JSON per CV in `data/ocr/`.

**Reading order.** Lines are grouped into visual rows (same y-centre within half a line height)
and read top-to-bottom, left-to-right. **Two-column layouts** are detected with an x-coverage
histogram of the boxes: if an (almost) empty vertical gutter exists in the central 20–80% of the
page with ≥20% of the lines on each side, the left column is read before the right one; lines that
cross the gutter (full-width headers) act as separators, so a job title stays next to its dates.
PopResume renders are single-column, so this branch is rarely triggered here, but it is tested and
kept for real CVs.

**Cleaning.** The key fix: **PaddleOCR drops the en dash in date ranges** (`2019.08–2023.01` is read
as `2019.082023.01`), so the cleaner re-inserts it. Also: de-hyphenation, unified dashes, whitespace
normalization, and removal of the name line, emails, URLs, phone numbers, the `ST, USA` location and
street addresses. Low-confidence lines are **kept** in the main setup (dropping them can delete a
date); a variant that drops lines with confidence < 0.80 is evaluated in Question 3.

**Split.** 70/15/15 by CV ID (1,400 / 300 / 300), stratified by occupation × experience band
(0–2, 3–5, 6–10, 11–20), seed 42 (`data/splits/`). All tuning uses the validation set; the test set
is used once at the end.

**Models.**
| Model | Setup |
|---|---|
| Median | Always predicts the training-set median (5.45 years). |
| TF-IDF + Ridge | Word 1–2-grams + char 3–5-grams, sublinear TF. **Digits and years kept** as tokens (they are the evidence; removing them is tested in Q3). Ridge α = 0.3, tuned on validation. |
| Embeddings + Ridge | Frozen `BAAI/bge-small-en-v1.5`. CVs longer than the encoder limit would be split into 250-word windows and averaged; Ridge α = 0.1, tuned on validation. |
| Fine-tuned BERT | `distilbert-base-uncased`, `num_labels=1`, `problem_type="regression"`, MSE loss, target ÷ 20 during training. Max length 512 with **head + tail** truncation (first 320 + last 190 tokens). In practice CVs are short (median 95 tokens, 90th percentile 136): **no CV exceeds 512 tokens and 100% of the years survive**. lr 5e-5, batch 16, up to 20 epochs, early stopping on validation MAE (patience 5), seed 42, fp16 on an NVIDIA RTX 5050 Laptop GPU (8 GB). Training time 93 s. |
| Rule-based (bonus) | Regex for `YYYY.MM–YYYY.MM` ranges, merge overlapping intervals, add up the covered years; no training. |

## Results

Test set, n = 300 (`output/tables/model_comparison.csv`).

| Model | MAE | RMSE | Acc ±1 | Acc ±2 | Acc ±3 | Band acc. | Band macro-F1 | R² | Train time | Infer./CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Median | 3.59 | 4.59 | 16.0% | 34.0% | 51.0% | 31.3% | 0.12 | -0.05 | 0 s | 0 ms |
| TF-IDF + Ridge | 1.01 | 1.39 | 61.3% | 88.7% | 96.3% | 81.7% | 0.82 | 0.90 | 9.5 s | 0.5 ms |
| Embeddings + Ridge | 1.31 | 1.70 | 45.7% | 79.0% | 94.3% | 73.0% | 0.73 | 0.86 | 2.2 s | 1.3 ms |
| **Fine-tuned BERT** | **0.31** | **0.40** | **98.0%** | **100%** | **100%** | **91.3%** | **0.92** | **0.99** | 93 s (GPU) | 1.2 ms (GPU) |
| Rule-based (bonus) | 0.02 | 0.04 | 100% | 100% | 100% | 99.0% | 0.99 | 1.00 | 0 s | <0.1 ms |

**BERT's MAE gain** (95% paired bootstrap CI, 2,000 resamples, `output/tables/bert_gain_bootstrap.csv`):

| vs | MAE reduction (years) | 95% CI | Significant |
|---|---|---|---|
| Median | 3.28 | [2.96, 3.60] | yes |
| TF-IDF + Ridge | 0.70 | [0.59, 0.80] | yes |
| Embeddings + Ridge | 0.99 | [0.88, 1.13] | yes |
| Rule-based (bonus) | −0.29 (BERT is worse) | [−0.32, −0.27] | yes |

**MAE by true experience band** (`output/tables/mae_by_band.csv`):

| Band | Median | TF-IDF | Embeddings | BERT | Rule |
|---|---|---|---|---|---|
| 0–2 | 4.17 | 0.95 | 1.17 | 0.29 | 0.02 |
| 3–5 | 1.55 | 0.73 | 1.18 | 0.29 | 0.03 |
| 6–10 | 2.40 | 0.96 | 1.29 | 0.32 | 0.02 |
| 11–20 | 8.49 | 1.64 | 1.71 | 0.36 | 0.02 |

**MAE by occupation** (BERT): accountants 0.26, construction 0.28, teachers 0.26, nurses 0.44,
software developers 0.34; TF-IDF ranges 0.85–1.23 (`output/tables/mae_by_occupation.csv`).

**Predicted vs true** (`output/figures/pred_vs_true.png`): the median is a flat line; TF-IDF and
embeddings show regression to the mean (juniors overestimated, seniors underestimated, MAE 1.64 in
the 11–20 band); BERT hugs the 45° line in every band; the rule sits on it.

**Error analysis — 10 largest BERT errors** (all between 0.9 and 1.3 years,
`output/tables/bert_top10_errors.csv`):
| Cause | Count | Pattern |
|---|---|---|
| Career gap | 5 | CVs with 1–2 years of gaps between jobs; BERT predicts closer to the full span (first start → last end) than to the sum of jobs, i.e. it does not fully subtract gaps. |
| Single short job | 2 | One 1–2-year job; BERT is pulled toward the mean (2.36 → 1.05, 1.33 → 2.41). |
| Long career underestimated | 2 | 4 jobs, 16 years; mild regression to the mean. |
| Other | 1 | 2 jobs with a 1.3-year gap, underestimated. |
| OCR misread / reading order / truncation / missing dates | 0 | After the dash repair, no error traces back to OCR. |

**Possible improvements:** feed BERT the parsed date ranges (or the rule's estimate) as an extra
input; augment training with CVs that have gaps; average several seeds; for real CVs, use the server
OCR models on GPU and a layout-aware model (LayoutLM) for varied date formats.

## Analytical questions

**Q1 — What does BERT learn that the baselines cannot?**
BERT cuts MAE by **0.70 years vs TF-IDF** [95% CI 0.59–0.80], **0.99 vs embeddings** [0.88–1.13]
and **3.28 vs the median** [2.96–3.60]; all significant. The gain grows with seniority: in the
11–20 band BERT's MAE is 0.36 vs 1.64 for TF-IDF (−1.28 years), against −0.44 to −0.65 in the
lower bands — bag-of-words models cannot add up four or five job durations, so they collapse long
careers toward the mean. **Probe** (`output/tables/probe_shift_years.csv`, 20 test CVs each):
(a) *delete the oldest job* — a recruiter would lower the estimate by that job's length (2.57 years
on average). BERT moves −2.79 years, every CV moves down, and its change correlates **0.90** with the
expected one; TF-IDF moves only −1.18 (correlation 0.57) and embeddings −2.31 (0.47). (b) *shift
every year by +5* — durations are unchanged, so a recruiter's estimate would not move. BERT still
moves 1.34 years on average, TF-IDF 1.96 and embeddings 2.44. **Conclusion:** BERT reads the dates
and does approximate date arithmetic — far more than the bag-of-words models — but it has also
partly memorized absolute years (e.g. "2023" as a frequent end date), and it only partly subtracts
career gaps. The rule does exact arithmetic and is invariant to the shift (0.00).

**Q2 — How much does OCR noise cost?**
OCR is very accurate on these renders: **mean CER 0.92%, mean WER 4.7%** (by occupation, CER
0.75–1.12%). Its one systematic failure is **dropping the en dash in date ranges**, so only
**81.9% of reference years are read as separate years** (29% of CVs fully correct; worst for nurses
and software developers, ~77%). The digits themselves are right: after re-inserting the dash,
**100% of years are recovered in 100% of CVs**. Evaluating each model on OCR text vs the reference
text of the same 300 test CVs (`output/tables/ocr_vs_reference.csv`): median 0.00, TF-IDF −0.01,
embeddings −0.03, rule +0.002, BERT −0.14 years (BERT is *better* on OCR text, because it was trained
on OCR text and the reference text's formatting differs slightly). With cleaning, OCR noise costs
nothing measurable. Without the dash repair, the cost is large for anything that parses dates: the
rule's MAE jumps from 0.02 to **3.60** (Q3). BERT is the most robust learned model to raw OCR
(0.23 on raw text) because sub-word tokens like `2019`, `.`, `08`, `2023` still carry the numbers
when the dash is missing.

**Q3 — Methodological sensitivity** (test MAE, `output/tables/sensitivity.csv`)
| Setup | Median | TF-IDF | Embeddings | BERT | Rule |
|---|---|---|---|---|---|
| Raw OCR (reading order only) | 3.59 | 1.04 | 1.58 | 0.23 | 3.60 |
| Cleaned OCR (main) | 3.59 | 1.01 | 1.31 | 0.31 | 0.02 |
| Cleaned, lines with confidence < 0.80 removed | 3.59 | 1.01 | 1.31 | 0.30 | 0.02 |

Other choices: **removing digits from TF-IDF** raises its MAE from 1.01 to **1.82** (years are the
signal); **BERT with Huber loss** 0.21 vs 0.31 with MSE; **head-only truncation** 0.22 vs 0.31 for
head + tail. Cleaning matters a lot for the rule and for embeddings (1.58 → 1.31), little for TF-IDF.
The confidence filter changes nothing (OCR confidence is ~0.99 everywhere). **Caveat:** since no CV
exceeds 512 tokens, head and head + tail feed BERT *identical* inputs — the 0.31 vs 0.22 difference
is pure run-to-run variance of fine-tuning (non-deterministic GPU kernels). BERT variants therefore
differ by up to ~0.1 years from noise alone, so the raw-vs-clean and MSE-vs-Huber differences for
BERT are not conclusive; a fair comparison would average several seeds.

**Q4 — Business value, fairness and ethics.**
*Who and for what:* job boards and recruiting agencies could **pre-fill the "years of experience"
field** of a profile from an uploaded CV, and HR teams could **sort or route** large queues for a
human reviewer (e.g. junior vs senior pipelines). Deciding who gets an interview is a different,
much riskier use and is out of scope. *Cost of errors:* ±2 years rarely changes a seniority level;
±5 years moves a candidate between junior and senior and can hide them from the right search.
**Underestimating is worse** in screening: it silently filters out qualified people, who never learn
why; an overestimate is usually caught at the interview. BERT is within ±2 years for 100% of test
CVs; TF-IDF misses by more than 2 years in 11%. *Fairness check* (`output/tables/fairness.csv`,
BERT on 300 test CVs):

| Group | n | MAE | Mean error (pred − true) |
|---|---|---|---|
| Female | 148 | 0.35 | +0.15 |
| Male | 152 | 0.28 | +0.01 |
| Asian / Pacific Islander | 33 | 0.26 | +0.09 |
| Black | 21 | 0.41 | +0.05 |
| White | 246 | 0.32 | +0.08 |

BERT slightly **overestimates women** (+0.15 years) and is a bit less accurate for them; Black
candidates have the highest MAE, but n = 21 is too small to conclude. No group is systematically
underestimated. Sex and race are not inputs, but they can leak through proxies: PopResume generates
careers from population statistics, so career gaps (linked to caregiving), occupation mix and
education paths differ by group — and gaps are exactly where BERT errs most (see error analysis).

## Limitations

Synthetic resumes generated from U.S. statistics with one clean template and one date format
(`YYYY.MM–YYYY.MM`) — which is why a regex can solve the task; real CVs are messier (free-text
dates, "Present", scans, multiple columns), where a learned model may be needed. Only 5 occupations;
English only; 2,000 CVs instead of 4,000; OCR on CPU with mobile models; a single seed per model
(BERT varies by ~0.1 years between runs). Such a system must **never decide who is hired**: it
estimates one number from a document, cannot see context (part-time work, career breaks, unpaid
work) and may carry group-level bias.

## Reproduce

```bash
conda create -n cvexp python=3.11 -y && conda activate cvexp
pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; omit the index for CPU
pip install -r requirements.txt
hf auth login                    # after accepting the PopResume access conditions

python src/download_data.py      # 400 no-photo CVs per occupation -> data/raw/
python src/ocr_extraction.py     # PaddleOCR -> data/ocr/ (cached, re-runnable)
python src/preprocessing.py      # reading order, cleaning, masking -> data/processed/
python src/ocr_quality.py        # CER / WER / years read correctly
python src/split.py              # 70/15/15 -> data/splits/
python src/baseline_median.py
python src/baseline_rules.py     # bonus
python src/baseline_tfidf.py
python src/baseline_tfidf.py --no-digits            # Q3
python src/baseline_embeddings.py
python src/train_bert.py                            # main model
python src/train_bert.py --variant raw              # Q3
python src/train_bert.py --variant clean_conf       # Q3
python src/train_bert.py --loss huber               # Q3
python src/train_bert.py --strategy head            # Q3
python src/evaluation.py         # -> output/tables/
python src/visualization.py      # -> output/figures/
```

Images (`data/raw/`) and model weights (`models/`) are not committed; the commands above regenerate
them. OCR JSONs, processed texts, splits, predictions, tables and figures are committed.

## Run the app

```bash
streamlit run app.py
```

Four tabs: Problem & Methodology · OCR & Exploratory Analysis · Model Results · Interactive Estimator
(upload a CV image or PDF, PaddleOCR runs, and all models predict side by side).

## Workflow

No direct commits to `main`: each stage was developed on a `feature/...` branch (data-sampling,
paddleocr-extraction, baselines, bert-regression, evaluation, streamlit-dashboard) and merged
through Pull Requests.
