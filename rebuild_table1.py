#!/usr/bin/env python3
r"""
Rebuild Table 1 from the RELEASED file (17,500 clips, 2,500/class) with REAL
durations read from WAVS_16K. Prints per-dialect stats, the corpus total (for
the abstract), and ready-to-paste LaTeX for Table 1.

Run in the folder with aydid_master.csv, or set the paths below.
"""

import os, csv, wave, contextlib
from collections import defaultdict

BASE       = r"F:\DID_ASR_Datasets\AYDID"
CSV_PATH   = os.path.join(BASE, "aydid_master.csv")
AUDIO_ROOT = os.path.join(BASE, "WAVS_16K")

REGION = {"YEM_AD": "Adeni", "YEM_BA": "Badawi", "YEM_HA": "Hadrami",
          "YEM_SA": "Sana'ani", "YEM_ST": "Standard", "YEM_TA": "Ta'izzi",
          "YEM_TI": "Tihami"}
ORDER = ["YEM_AD", "YEM_BA", "YEM_HA", "YEM_SA", "YEM_ST", "YEM_TA", "YEM_TI"]

BS = "\\"          # backslash, kept out of f-strings
ROWEND = BS + BS   # LaTeX row terminator '\\'

def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return [{k.lstrip("\ufeff"): v for k, v in r.items()} for r in csv.DictReader(f)]

def resolve(r):
    p1 = os.path.join(AUDIO_ROOT, *r["file_path"].split("/"))
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(AUDIO_ROOT, r["wav_name"])
    return p2 if os.path.exists(p2) else None

def dur_seconds(path):
    with contextlib.closing(wave.open(path, "r")) as w:
        return w.getnframes() / float(w.getframerate())

rows = load_rows(CSV_PATH)
sec = defaultdict(float); utt = defaultdict(int); spk = defaultdict(set)
missing = 0
for i, r in enumerate(rows):
    d = r["dialect"]
    utt[d] += 1
    spk[d].add(r["speaker_id"])
    p = resolve(r)
    if p is None:
        missing += 1
        continue
    try:
        sec[d] += dur_seconds(p)
    except Exception:
        missing += 1
    if i % 2000 == 0:
        print(f"  {i}/{len(rows)}")

if missing:
    print(f"WARNING: {missing} files unreadable/missing; durations undercount.")

print("\n--- per-dialect ---")
tot_utt = 0; tot_sec = 0.0; tot_spk = 0
for d in ORDER:
    h = sec[d] / 3600.0
    print(f"{d} {REGION[d]:9s} spk={len(spk[d])} utt={utt[d]:5d} dur={h:5.2f} h")
    tot_utt += utt[d]; tot_sec += sec[d]; tot_spk += len(spk[d])
tot_h = tot_sec / 3600.0
print(f"TOTAL spk={tot_spk} utt={tot_utt} dur={tot_h:.2f} h")

# ---- LaTeX for Table 1 ----
print("\n--- paste this as Table 1 ---\n")
header = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{AYDID statistics by sub-dialect. Every class is balanced at 50 "
    r"speakers and 2,500 utterances (50 per speaker) by design.}",
    r"\label{tab:corpus}",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Dialect & Region & Spk. & Utt. & Dur.\,(h) " + ROWEND,
    r"\midrule",
]
print("\n".join(header))
for d in ORDER:
    h = sec[d] / 3600.0
    label = d.replace("_", BS + "_")
    utt_s = f"{utt[d]:,}"
    print(f"{label} & {REGION[d]} & {len(spk[d])} & {utt_s} & {h:.2f} " + ROWEND)
print(r"\midrule")
tot_utt_s = f"{tot_utt:,}"
print(f"{BS}textbf{{Total}} & --- & {BS}textbf{{{tot_spk}}} & "
      f"{BS}textbf{{{tot_utt_s}}} & {BS}textbf{{{tot_h:.2f}}} " + ROWEND)
print("\n".join([r"\bottomrule", r"\end{tabular}", r"\end{table}"]))

print(f"\n>>> Update abstract/intro: '{tot_h:.2f} hours', '{tot_utt_s} utterances'.")