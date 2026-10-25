"""Stage 5 - Fine-tune DistilBERT as a regressor (num_labels=1, problem_type="regression").

- Target scaled by /CAP during training; predictions converted back to years and clipped to 0-CAP.
- Long CVs: head + tail truncation (first HEAD_TOKENS + last tokens up to MAX_LEN). The head holds
  the summary and most recent jobs, the tail holds the oldest jobs and education dates, which
  together bound the career. The share of years that survive truncation is logged.
- Early stopping on validation MAE (in years).

Usage:
    python src/train_bert.py                                  # main run: clean text, MSE, head+tail
    python src/train_bert.py --variant raw                    # Question 3 (OCR cleaning)
    python src/train_bert.py --loss huber                     # Question 3 (loss)
    python src/train_bert.py --strategy head                  # Question 3 (long-text strategy)
"""
import argparse
import copy
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from utils import CAP, EVAL_SETS, MAIN_VARIANT, MODELS, TARGET, ensure_dirs, load_sets, save_predictions, set_seed

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 512
HEAD_TOKENS = 320
HP = dict(lr=5e-5, batch_size=16, epochs=20, patience=5, warmup=0.1, weight_decay=0.01)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def encode(tok, texts, strategy: str) -> list:
    """Token ids per CV with the chosen long-text strategy (special tokens included)."""
    body = MAX_LEN - 2
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) > body:
            ids = ids[:body] if strategy == "head" else ids[:HEAD_TOKENS] + ids[-(body - HEAD_TOKENS):]
        out.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    return out


def truncation_report(tok, texts, encoded) -> dict:
    lens = [len(tok(t, add_special_tokens=False, truncation=False)["input_ids"]) for t in texts]
    kept = []
    for t, ids in zip(texts, encoded):
        years = YEAR.findall(t)
        if years:
            surv = set(YEAR.findall(tok.decode(ids)))
            kept.append(np.mean([y in surv for y in years]))
    return {"share_longer_than_max_len": float(np.mean(np.array(lens) > MAX_LEN - 2)),
            "tokens_median": float(np.median(lens)), "tokens_p90": float(np.percentile(lens, 90)),
            "share_years_surviving_truncation": float(np.mean(kept)) if kept else None}


def collate(tok):
    def fn(batch):
        ids = [b[0] for b in batch]
        enc = tok.pad({"input_ids": ids}, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.float)
        return enc
    return fn


def predict(model, tok, texts, strategy: str, device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    enc = encode(tok, texts, strategy)
    loader = DataLoader([(e, 0.0) for e in enc], batch_size=batch_size, collate_fn=collate(tok))
    out = []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items() if k != "labels"}
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out.append(model(**b).logits.squeeze(-1).float().cpu().numpy())
    return np.clip(np.concatenate(out) * CAP, 0, CAP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=MAIN_VARIANT)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--strategy", choices=["head_tail", "head"], default="head_tail")
    args = parser.parse_args()
    set_seed()
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sets = load_sets(args.variant)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, problem_type="regression").to(device)

    tr_enc = encode(tok, sets["train"]["text"].tolist(), args.strategy)
    trunc = truncation_report(tok, sets["train"]["text"].tolist(), tr_enc)
    train_data = list(zip(tr_enc, (sets["train"][TARGET].values / CAP).tolist()))
    g = torch.Generator().manual_seed(42)
    loader = DataLoader(train_data, batch_size=HP["batch_size"], shuffle=True, generator=g,
                        collate_fn=collate(tok))
    loss_fn = torch.nn.MSELoss() if args.loss == "mse" else torch.nn.HuberLoss(delta=0.1)  # 0.1 = 2 years
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    steps = len(loader) * HP["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(HP["warmup"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_mae, best_state, bad, history = float("inf"), None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            labels = b.pop("labels")
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**b).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        val_pred = predict(model, tok, sets["val"]["text"].tolist(), args.strategy, device)
        mae = float(np.mean(np.abs(val_pred - sets["val"][TARGET].values)))
        history.append({"epoch": epoch, "val_mae": mae})
        print(f"epoch {epoch}: val MAE = {mae:.3f}", flush=True)
        if mae < best_mae:
            best_mae, best_state, bad = mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    t0 = time.perf_counter()
    preds = {name: predict(model, tok, sets[name]["text"].tolist(), args.strategy, device)
             for name in EVAL_SETS}
    infer = (time.perf_counter() - t0) / sum(len(sets[n]) for n in preds)

    tag = "bert" + ("" if args.loss == "mse" else "_huber") + ("" if args.strategy == "head_tail" else "_head")
    save_predictions(tag, args.variant, sets, preds, {
        "train_seconds": train_s, "infer_seconds_per_cv": infer, "model": MODEL_NAME,
        "max_len": MAX_LEN, "head_tokens": HEAD_TOKENS, "strategy": args.strategy, "loss": args.loss,
        "seed": 42, **HP, "best_val_mae": best_mae, "history": history, "truncation": trunc,
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    })
    if tag == "bert" and args.variant == MAIN_VARIANT:
        out = MODELS / "bert" / "best"
        model.save_pretrained(out)
        tok.save_pretrained(out)
    print(f"{tag} [{args.variant}] best val MAE = {best_mae:.3f}, trained in {train_s / 60:.1f} min")


if __name__ == "__main__":
    main()
