#!/usr/bin/env python3
r"""
Assemble the AYDID Hugging Face release from the master CSV and the frozen
embeddings produced by extract_embeddings.py.

Produces (under hf_release/):
  embeddings.npy            (N x 1024 float32, row order == metadata.csv)
  metadata.csv              wav_name,dialect,speaker_id,show_name,content_type,
                            gender,age,transcript,has_transcript
  splits/speaker_disjoint.json
  splits/source_disjoint.json   (4 leave-one-programme-per-dialect-out folds)
  reproduce_probe.py        standalone: reproduces 75.3 -> 74.6 from the release
  README.md                 dataset card
  LICENSE.txt               CC BY-NC-SA note
  checkpoints_manifest.json slot for the Whisper-Yemeni + MMS-300m checkpoints
  manifest.json             counts + sha256 for integrity

Split logic is IDENTICAL to run_did_from_embeddings.py (seed 42), so the
released splits reproduce the paper's probe numbers exactly.

Prereqs:
  extract_embeddings.py already run -> release/embeddings.npy + release/index.json
"""

import os, csv, json, shutil, hashlib
import numpy as np
from collections import defaultdict

# ============================ CONFIG ============================
BASE      = r"F:\DID_ASR_Datasets\AYDID"
CSV_PATH  = os.path.join(BASE, "aydid_master.csv")
EMB_NPY   = os.path.join(BASE, "release", "embeddings.npy")   # from extract_embeddings.py
EMB_INDEX = os.path.join(BASE, "release", "index.json")
OUT_DIR   = os.path.join(BASE, "hf_release")
SEED      = 42
CITATION  = ("@inproceedings{aydid2026,\n"
             "  title     = {AYDID: A Sub-Dialectal Yemeni Arabic Corpus for "
             "Dialect Identification and Speech Recognition},\n"
             "  author    = {Ba Mahel, Mansoor S. M. and Wei, Jianguo and Yue, "
             "Xianghu and Awn, Norah Saeed and Bamahel, Abdulaziz S.},\n"
             "  booktitle = {Proc. IEEE ICASSP},\n"
             "  year      = {2026}\n}")
# ===============================================================

def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return [{k.lstrip("\ufeff"): v for k, v in r.items()} for r in csv.DictReader(f)]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------- load metadata + embeddings, align by wav_name ----------
rows = load_rows(CSV_PATH)
order = [r["wav_name"] for r in rows]

emb = np.load(EMB_NPY)
idx = json.load(open(EMB_INDEX, encoding="utf-8"))
emb_by_name = {idx[i]["wav_name"]: emb[i] for i in range(len(idx))}
missing = [w for w in order if w not in emb_by_name]
if missing:
    raise SystemExit(f"{len(missing)} clips have no embedding (first: {missing[:3]}). "
                     f"Re-run extract_embeddings.py.")
emb_ordered = np.stack([emb_by_name[w] for w in order]).astype(np.float32)

os.makedirs(os.path.join(OUT_DIR, "splits"), exist_ok=True)

# ---------- metadata.csv (release copy) ----------
meta_cols = ["wav_name", "dialect", "speaker_id", "show_name", "content_type",
             "gender", "age", "transcript", "has_transcript"]
with open(os.path.join(OUT_DIR, "metadata.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=meta_cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in meta_cols})

# ---------- embeddings.npy (aligned to metadata row order) ----------
np.save(os.path.join(OUT_DIR, "embeddings.npy"), emb_ordered)

# ---------- splits (identical logic to run_did_from_embeddings.py) ----------
y    = np.array([r["dialect"] for r in rows])
spk  = np.array([r["speaker_id"] for r in rows])
show = np.array([r["show_name"] for r in rows])
name = np.array(order)
labels = sorted(set(y.tolist()))

# speaker-disjoint: ~20% speakers per dialect held out
rng = np.random.RandomState(SEED)
dia_spk = defaultdict(set)
for i in range(len(rows)):
    dia_spk[y[i]].add(spk[i])
test_spk = set()
for d, sp in dia_spk.items():
    sp = sorted(sp); rng.shuffle(sp)
    test_spk.update(sp[: max(1, round(0.2 * len(sp)))])
sd = {
    "description": "Speaker-disjoint: ~20% of speakers per dialect held out (seed 42).",
    "train": [order[i] for i in range(len(rows)) if spk[i] not in test_spk],
    "test":  [order[i] for i in range(len(rows)) if spk[i] in test_spk],
}
json.dump(sd, open(os.path.join(OUT_DIR, "splits", "speaker_disjoint.json"), "w"),
          ensure_ascii=False, indent=1)

# source-disjoint: leave-one-programme-per-dialect-out, 4 folds
dia_shows = defaultdict(list)
for i in range(len(rows)):
    if show[i] not in dia_shows[y[i]]:
        dia_shows[y[i]].append(show[i])
for d in dia_shows:
    dia_shows[d] = sorted(dia_shows[d])
n_folds = min(len(v) for v in dia_shows.values())
folds = []
for f in range(n_folds):
    heldout = {d: dia_shows[d][f % len(dia_shows[d])] for d in dia_shows}
    ho = set(heldout.values())
    folds.append({
        "fold": f + 1,
        "heldout_shows": heldout,
        "train": [order[i] for i in range(len(rows)) if show[i] not in ho],
        "test":  [order[i] for i in range(len(rows)) if show[i] in ho],
    })
json.dump({"description": "Source-disjoint: leave-one-programme-per-dialect-out, "
                          f"{n_folds} folds. Reproduces the paper's 74.6 wF1.",
           "n_folds": n_folds, "folds": folds},
          open(os.path.join(OUT_DIR, "splits", "source_disjoint.json"), "w"),
          ensure_ascii=False, indent=1)

# ---------- standalone reproduction script ----------
REPRO = r'''#!/usr/bin/env python3
"""Reproduce the AYDID source-robustness probe (paper Sec. 5.4) from this release.
Expected: speaker-disjoint ~75.3, source-disjoint ~74.6, leakage ~0.7 pp."""
import json, csv, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

rows = list(csv.DictReader(open("metadata.csv", encoding="utf-8")))
name2i = {r["wav_name"]: i for i, r in enumerate(rows)}
y = np.array([r["dialect"] for r in rows])
X = np.load("embeddings.npy")

def probe(tr_names, te_names):
    tr = [name2i[n] for n in tr_names]; te = [name2i[n] for n in te_names]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    return f1_score(y[te], pred, average="weighted") * 100

sd = json.load(open("splits/speaker_disjoint.json", encoding="utf-8"))
f_spk = probe(sd["train"], sd["test"])
print(f"speaker-disjoint wF1 = {f_spk:.1f}")

src = json.load(open("splits/source_disjoint.json", encoding="utf-8"))
fs = [probe(fo["train"], fo["test"]) for fo in src["folds"]]
f_src = float(np.mean(fs))
print(f"source-disjoint wF1 = {f_src:.1f} (sd {np.std(fs):.1f})")
print(f"leakage delta = {f_spk - f_src:.1f} pp")
'''
open(os.path.join(OUT_DIR, "reproduce_probe.py"), "w", encoding="utf-8").write(REPRO)

# ---------- dataset card ----------
n = len(rows)
hrs = None  # durations live in the paper; not recomputed here
card = f"""---
license: cc-by-nc-sa-4.0
task_categories:
- audio-classification
- automatic-speech-recognition
language:
- ar
tags:
- yemeni-arabic
- dialect-identification
- sub-dialectal
pretty_name: AYDID
---

# AYDID: Sub-Dialectal Yemeni Arabic Corpus

First speech corpus to model Yemeni Arabic at the **sub-dialectal** level:
{n:,} utterances, 350 speakers, 7 classes (Adeni, Badawi, Hadrami, Sana'ani,
Ta'izzi, Tihami, Standard Yemeni), balanced at 50 speakers / 2,500 utterances
per class.

## What this release contains

To respect the copyright of the broadcast source material, **audio is not
redistributed.** Instead we release:

- `embeddings.npy` — frozen mean-pooled MMS-300m embeddings, one 1024-d vector
  per utterance, row order matching `metadata.csv`.
- `metadata.csv` — dialect, speaker, source programme, content type, gender,
  age, and transcription for every utterance.
- `splits/` — the **speaker-disjoint** and **source-disjoint**
  (leave-one-programme-per-dialect-out, 4 folds) split definitions used in the
  paper's source-robustness analysis.
- `reproduce_probe.py` — reproduces the paper's probe: speaker-disjoint ~75.3
  vs. source-disjoint ~74.6 weighted F1 (leakage ~0.7 pp), directly from the
  files above.
- Fine-tuned **checkpoints** (Whisper-Yemeni ASR; MMS-300m DID) are released
  separately; see `checkpoints_manifest.json`.

## Reproduce the source-robustness result

```bash
python reproduce_probe.py
# speaker-disjoint wF1 = 75.3
# source-disjoint  wF1 = 74.6
# leakage delta        = 0.7 pp
```

## Citation

```
{CITATION}
```

Licensed CC BY-NC-SA 4.0 — research use, with attribution.
"""
open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8").write(card)

# ---------- license + checkpoint manifest ----------
open(os.path.join(OUT_DIR, "LICENSE.txt"), "w", encoding="utf-8").write(
    "AYDID derived features, transcriptions, and metadata are released under\n"
    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).\n"
    "Source audio is NOT redistributed; copyright remains with the original\n"
    "broadcasters. Research use only, with attribution to the paper.\n")

json.dump({
    "note": "Upload the two fine-tuned checkpoints and fill in the URLs.",
    "checkpoints": [
        {"name": "whisper-yemeni", "base": "openai/whisper-large-v2",
         "task": "ASR", "reproduces": "Table (WER+ 19.14)", "url": "TODO"},
        {"name": "mms-300m-aydid-did", "base": "badrex/mms-300m-arabic-dialect-identifier",
         "task": "DID", "reproduces": "Table (weighted F1 80.44)", "url": "TODO"},
    ],
}, open(os.path.join(OUT_DIR, "checkpoints_manifest.json"), "w"), indent=2)

# ---------- integrity manifest ----------
emb_path = os.path.join(OUT_DIR, "embeddings.npy")
json.dump({
    "n_utterances": n,
    "n_speakers": len(set(spk.tolist())),
    "n_programmes": len(set(show.tolist())),
    "classes": labels,
    "embedding_dim": int(emb_ordered.shape[1]),
    "embeddings_sha256": sha256(emb_path),
    "source_disjoint_folds": n_folds,
    "speaker_disjoint_test_n": len(sd["test"]),
}, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)

print(f"Release written to {OUT_DIR}")
print(f"  embeddings.npy  {emb_ordered.shape}")
print(f"  metadata.csv    {n} rows")
print(f"  splits/         speaker_disjoint (test n={len(sd['test'])}), "
      f"source_disjoint ({n_folds} folds)")
print("Next: cd hf_release && python reproduce_probe.py  (sanity check), "
      "then upload to Hugging Face.")
