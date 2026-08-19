#!/usr/bin/env python3
r"""
Render the fine-tuned DID confusion matrix as a single-column ICASSP figure.

USAGE:
  1. Paste your real 7x7 counts into COUNTS below (rows = TRUE label, in the
     order AD, BA, HA, SA, ST, TA, TI; each row must sum to 375).
     These are the counts behind your original Figure 3 (fine-tuned MMS-300m).
  2. python make_confusion_fig.py
  3. It writes confusion_matrix.pdf next to Template.tex; recompile the paper.

The row-sum assert will stop you if a row != 375, so a half-filled matrix
cannot slip through.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["AD", "BA", "HA", "SA", "ST", "TA", "TI"]

# ----- REPLACE THESE WITH YOUR REAL COUNTS (rows sum to 375) -----
# Row order = TRUE dialect; column order = PREDICTED dialect.
# The only cell known from the manuscript is HA->BA = 119; everything else
# below is a PLACEHOLDER and must be overwritten with your Figure-3 numbers.
COUNTS = [
    #  AD   BA   HA   SA   ST   TA   TI   (predicted ->)
    [   0,   0,   0,   0,   0,   0,   0],  # AD true  -> fill (sum 375)
    [   0,   0,   0,   0,   0,   0,   0],  # BA true
    [   0, 119,   0,   0,   0,   0,   0],  # HA true  (HA->BA = 119 known)
    [   0,   0,   0,   0,   0,   0,   0],  # SA true
    [   0,   0,   0,   0,   0,   0,   0],  # ST true
    [   0,   0,   0,   0,   0,   0,   0],  # TA true
    [   0,   0,   0,   0,   0,   0,   0],  # TI true
]
# -----------------------------------------------------------------

def render(counts, out="confusion_matrix.pdf"):
    M = np.array(counts, dtype=int)
    assert M.shape == (7, 7), "matrix must be 7x7"
    rs = M.sum(axis=1)
    bad = [(LABELS[i], int(rs[i])) for i in range(7) if rs[i] != 375]
    if bad:
        raise SystemExit(f"Row sums must all be 375. Offending rows: {bad}\n"
                         f"-> paste your real counts into COUNTS.")

    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    im = ax.imshow(M, cmap="viridis")           # viridis reads fine in grayscale
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    thr = M.max() / 2.0
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < thr else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

if __name__ == "__main__":
    render(COUNTS)
