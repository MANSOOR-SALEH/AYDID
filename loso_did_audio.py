#!/usr/bin/env python3
r"""
Show-disjoint (leave-one-show-out) DID for AYDID  -- audio, MMS-300m.

Produces the SOURCE-DISJOINT weighted-F1 that sits next to the paper's
speaker-disjoint 80.44. Fold logic matches the text-proxy run exactly, so the
audio and text numbers are directly comparable.

DATA LAYOUT (yours):
    D:\Paper_1_ICASSP\... (this script)
    F:\DID_ASR_Datasets\AYDID\aydid_master.csv
    F:\DID_ASR_Datasets\AYDID\WAVS_16K\...        (16 kHz, already normalized)

The CSV stores file_path as  YEM_AD/YEM_AD_F001/YEM_AD_F001_001.wav
and wav_name as              YEM_AD_F001_001.wav
The script tries WAVS_16K/<file_path> first, then WAVS_16K/<wav_name>.

RUN:
    pip install "accelerate>=1.1.0"      # required by Trainer
    pip install -r requirements.txt
    Run this file in PyCharm.  MODE="single" = 1 retrain (cheap),
    MODE="cv" = 4 folds (the number to report).

The classifier MISMATCH warning (ckpt [5] vs model [7]) is EXPECTED: the base
model ships a 5-class MADIS-5 head; we replace it with a fresh 7-class head for
the Yemeni sub-dialects. Encoder weights load fine; only the final layer is new.
"""

import os, csv, json, numpy as np, torch, soundfile as sf
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          TrainingArguments, Trainer)

# ============================ CONFIG ============================
BASE       = r"F:\DID_ASR_Datasets\AYDID"
CSV_PATH   = os.path.join(BASE, "aydid_master.csv")   # <-- change if named differently
AUDIO_ROOT = os.path.join(BASE, "WAVS_16K")
MODEL_ID   = "badrex/mms-300m-arabic-dialect-identifier"
MODE       = "cv"        # "single" or "cv"
EPOCHS     = 5
BATCH      = 8           # lower to 4 if you hit CUDA OOM
LR         = 3e-5
SR         = 16000
MAX_SEC    = 10          # truncate very long clips (yours avg ~4.5 s)
SEED       = 42
OUT_DIR    = os.path.join(BASE, "loso_runs")
# ===============================================================

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------- load metadata ----------
def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:          # utf-8-sig strips BOM
        return [ {k.lstrip("\ufeff"): v for k, v in r.items()}
                 for r in csv.DictReader(f) ]

rows   = load_rows(CSV_PATH)
labels = sorted({r["dialect"] for r in rows})
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
print(f"clips={len(rows)}  labels={labels}")

def resolve(r):
    p1 = os.path.join(AUDIO_ROOT, *r["file_path"].split("/"))  # nested
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(AUDIO_ROOT, r["wav_name"])               # flat fallback
    return p2 if os.path.exists(p2) else None

# ---------- preflight: verify audio exists BEFORE training ----------
missing = 0
for r in rows:
    if resolve(r) is None:
        missing += 1
        if missing <= 5:
            print("  MISSING:", r["file_path"])
if missing:
    raise SystemExit(f"\n{missing} audio files not found under {AUDIO_ROOT}. "
                     f"Fix AUDIO_ROOT / layout, then rerun.")
print("preflight OK: all audio present.")

# ---------- shows per dialect (each show is single-dialect) ----------
dia_shows = defaultdict(list)
for r in rows:
    if r["show_name"] not in dia_shows[r["dialect"]]:
        dia_shows[r["dialect"]].append(r["show_name"])
for d in dia_shows:
    dia_shows[d] = sorted(dia_shows[d])
n_folds = min(len(v) for v in dia_shows.values())   # = 4 (Tihami)
print("shows/dialect:", {d: len(v) for d, v in sorted(dia_shows.items())},
      f"-> {n_folds} folds")

feat = AutoFeatureExtractor.from_pretrained(MODEL_ID)

# ---------- torch Dataset + dynamic-padding collator ----------
class AudioDS(torch.utils.data.Dataset):
    def __init__(self, idxs):
        self.idxs = idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        r = rows[self.idxs[i]]
        wav, sr = sf.read(resolve(r), dtype="float32")
        if wav.ndim > 1:                      # stereo -> mono
            wav = wav.mean(axis=1)
        if sr != SR:                          # should already be 16k
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        wav = wav[: SR * MAX_SEC]
        return {"wav": wav, "label": lab2id[r["dialect"]]}

class Collator:
    def __init__(self, fe): self.fe = fe
    def __call__(self, batch):
        wavs   = [b["wav"] for b in batch]
        labels = [int(b["label"]) for b in batch]
        out = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out

def metrics(p):
    pred = np.argmax(p.predictions, axis=1)
    return {"acc": accuracy_score(p.label_ids, pred),
            "wf1": f1_score(p.label_ids, pred, average="weighted")}

def run_fold(train_idx, test_idx, tag):
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID, num_labels=len(labels), id2label=id2lab, label2id=lab2id,
        ignore_mismatched_sizes=True)
    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, tag),
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS, learning_rate=LR,
        eval_strategy="epoch", save_strategy="no", logging_steps=50,
        seed=SEED, report_to="none",
        fp16=torch.cuda.is_available(), dataloader_num_workers=0,  # 0 = safe on Windows
        remove_unused_columns=False)          # keep 'wav'; our collator needs it
    trainer = Trainer(model=model, args=args,
                      train_dataset=AudioDS(train_idx), eval_dataset=AudioDS(test_idx),
                      data_collator=Collator(feat), compute_metrics=metrics)
    trainer.train()
    out  = trainer.predict(AudioDS(test_idx))
    pred = np.argmax(out.predictions, axis=1)
    wf1  = f1_score(out.label_ids, pred, average="weighted") * 100
    acc  = accuracy_score(out.label_ids, pred) * 100
    per  = {id2lab[i]: round(v * 100, 2)
            for i, v in enumerate(f1_score(out.label_ids, pred, average=None))}
    cm   = confusion_matrix(out.label_ids, pred).tolist()
    print(f"[{tag}] acc={acc:.2f}  wF1={wf1:.2f}  per-class={per}")
    json.dump({"acc": acc, "wf1": wf1, "per_class_f1": per,
               "labels": labels, "confusion": cm},
              open(os.path.join(OUT_DIR, f"{tag}.json"), "w"), indent=2)
    return acc, wf1

def fold_indices(fold):
    heldout = {d: dia_shows[d][fold % len(dia_shows[d])] for d in dia_shows}
    ho = set(heldout.values())
    te = [i for i, r in enumerate(rows) if r["show_name"] in ho]
    tr = [i for i, r in enumerate(rows) if r["show_name"] not in ho]
    return tr, te, heldout

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if MODE == "single":
        tr, te, ho = fold_indices(0)
        print("held-out shows:", ho, f"| train={len(tr)} test={len(te)}")
        run_fold(tr, te, "single")
    else:
        accs, wf1s = [], []
        for f in range(n_folds):
            tr, te, ho = fold_indices(f)
            print(f"\n=== fold {f+1}/{n_folds}  held-out shows: {ho} "
                  f"| train={len(tr)} test={len(te)} ===")
            a, w = run_fold(tr, te, f"fold{f+1}")
            accs.append(a); wf1s.append(w)
        summary = {"mode": "cv", "n_folds": n_folds,
                   "acc_mean": float(np.mean(accs)),
                   "wf1_mean": float(np.mean(wf1s)),
                   "wf1_sd": float(np.std(wf1s)),
                   "wf1_folds": [round(x, 2) for x in wf1s]}
        json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
        print(f"\n=== SHOW-DISJOINT (audio)  wF1 = {np.mean(wf1s):.2f} "
              f"(sd {np.std(wf1s):.2f})  |  report next to speaker-disjoint 80.44 ===")