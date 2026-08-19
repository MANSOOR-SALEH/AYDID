# AYDID: Sub-Dialectal Yemeni Arabic Corpus

Code for **AYDID**, the first sub-dialectal Yemeni Arabic speech corpus, with ASR
and dialect-identification (DID) benchmarks and a **source-disjoint** robustness
control. Companion to the ICASSP 2026 paper.

- **Dataset (features + metadata + splits):** https://huggingface.co/datasets/mansoorSaleh/AYDID
- **ASR checkpoint:** https://huggingface.co/mansoorSaleh/whisper-yemeni
- **DID checkpoint:** https://huggingface.co/mansoorSaleh/mms-300m-aydid-did

The corpus is 350 speakers across 7 classes (Adeni, Badawi, Hadrami, Sana'ani,
Ta'izzi, Tihami, Standard Yemeni), balanced at 50 speakers / 2,500 utterances per
class. To respect broadcast-media copyright, audio is **not** redistributed; the
release ships frozen MMS-300m embeddings, transcriptions, metadata, split
definitions, and the two fine-tuned checkpoints.

## Pipeline

Run in order. Scripts read/write under a base folder set in each file's `CONFIG`
block (default `F:\DID_ASR_Datasets\AYDID`).

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `extract_embeddings.py` | One forward pass over all clips → `release/embeddings.npy` (17,500 × 1024) + `index.json`. |
| 2 | `run_did_from_embeddings.py` | Trains a linear probe under **speaker-disjoint** vs **source-disjoint** (leave-one-programme-per-dialect-out, 4 folds); writes `summary.json`. Reproduces 75.3 → 74.6 (Δ 0.7 pp). |
| 3 | `make_confusion_fig.py` | Renders the fine-tuned DID confusion matrix (Fig. 1) → `confusion_matrix.pdf`. |
| 4 | `rebuild_table1.py` | Reads the WAVs, computes real per-dialect durations, prints Table 1 LaTeX. |
| 5 | `export_features.py` | Assembles the Hugging Face release bundle (embeddings, metadata, splits, dataset card, reproduction script). |

## Reproduce the source-robustness result

After downloading the dataset from Hugging Face:

```bash
cd AYDID            # the downloaded dataset folder
python reproduce_probe.py
# speaker-disjoint wF1 = 75.3
# source-disjoint  wF1 = 74.6
# leakage delta        = 0.7 pp
```

## Setup

Python 3.10–3.12. Install **torch with your CUDA** first (pick your build at
[pytorch.org](https://pytorch.org)), then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # example: CUDA 12.1
pip install -r requirements.txt
```

`extract_embeddings.py` needs a GPU; steps 2–5 run on CPU.

## Citation

```bibtex
@inproceedings{aydid2026,
  title     = {AYDID: A Sub-Dialectal Yemeni Arabic Corpus for Dialect
               Identification and Speech Recognition},
  author    = {Ba Mahel, Mansoor S. M. and Wei, Jianguo and Yue, Xianghu and
               Awn, Norah Saeed and Bamahel, Abdulaziz S.},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2026}
}
```

## License

Code: MIT. Dataset and derived features: CC BY-NC-SA 4.0 (research use). Source
audio is not redistributed; copyright remains with the original broadcasters.
