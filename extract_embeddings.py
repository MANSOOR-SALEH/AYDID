#!/usr/bin/env python3
r"""
STEP 1 of 2 -- extract frozen MMS-300m embeddings for every AYDID clip.

Forward pass only (no training) => no OOM, runs on a modest GPU in ~20-40 min.
Output doubles as the Hugging Face feature release:
    F:\DID_ASR_Datasets\AYDID\release\embeddings.npy   (N x 1024, float32)
    F:\DID_ASR_Datasets\AYDID\release\index.json        (row order = wav_name + labels)

Then run  run_did_from_embeddings.py  to get the speaker- vs show-disjoint numbers.
"""

import os, csv, json, numpy as np, torch, soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel

# ============================ CONFIG ============================
BASE       = r"F:\DID_ASR_Datasets\AYDID"
CSV_PATH   = os.path.join(BASE, "aydid_master.csv")
AUDIO_ROOT = os.path.join(BASE, "WAVS_16K")
MODEL_ID   = "badrex/mms-300m-arabic-dialect-identifier"
OUT_DIR    = os.path.join(BASE, "release")
SR         = 16000
MAX_SEC    = 8            # cap clip length fed to the encoder
EMB_BATCH  = 8            # forward-only; raise to 16/32 if VRAM allows
# ===============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return [{k.lstrip("\ufeff"): v for k, v in r.items()} for r in csv.DictReader(f)]

rows = load_rows(CSV_PATH)
print(f"clips={len(rows)}  device={device}")

def resolve(r):
    p1 = os.path.join(AUDIO_ROOT, *r["file_path"].split("/"))
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(AUDIO_ROOT, r["wav_name"])
    return p2 if os.path.exists(p2) else None

# preflight
missing = sum(resolve(r) is None for r in rows)
if missing:
    raise SystemExit(f"{missing} audio files missing under {AUDIO_ROOT}.")
print("preflight OK.")

feat = AutoFeatureExtractor.from_pretrained(MODEL_ID)
base = AutoModel.from_pretrained(MODEL_ID).to(device).eval()   # Wav2Vec2Model encoder
H = base.config.hidden_size
print(f"hidden size = {H}")

def load_wav(r):
    wav, sr = sf.read(resolve(r), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav[: SR * MAX_SEC]

embs = np.zeros((len(rows), H), dtype=np.float32)
use_amp = (device == "cuda")

for start in range(0, len(rows), EMB_BATCH):
    batch = rows[start:start + EMB_BATCH]
    wavs = [load_wav(r) for r in batch]
    inp = feat(wavs, sampling_rate=SR, return_tensors="pt",
               padding=True, return_attention_mask=True)
    iv = inp["input_values"].to(device)
    am = inp["attention_mask"].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
        out = base(input_values=iv, attention_mask=am)
        hs = out.last_hidden_state                      # (B, T, H)
        # masked mean-pool over time using the encoder's frame-level mask
        try:
            fmask = base._get_feature_vector_attention_mask(hs.shape[1], am)
        except Exception:
            fmask = torch.ones(hs.shape[:2], device=device, dtype=torch.long)
        fmask = fmask.unsqueeze(-1).float()
        pooled = (hs * fmask).sum(1) / fmask.sum(1).clamp(min=1)
    embs[start:start + len(batch)] = pooled.float().cpu().numpy()
    if (start // EMB_BATCH) % 50 == 0:
        print(f"  {start + len(batch)}/{len(rows)}")

np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs)
index = [{"wav_name": r["wav_name"], "dialect": r["dialect"],
          "speaker_id": r["speaker_id"], "show_name": r["show_name"],
          "content_type": r["content_type"], "gender": r["gender"],
          "age": r["age"]} for r in rows]
json.dump(index, open(os.path.join(OUT_DIR, "index.json"), "w"), ensure_ascii=False, indent=2)
print(f"\nsaved embeddings.npy  shape={embs.shape}  and index.json  ->  {OUT_DIR}")
print("next: run  run_did_from_embeddings.py")
