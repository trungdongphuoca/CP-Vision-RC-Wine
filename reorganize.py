"""
reorganize.py
=============
Tự động sắp xếp lại thư mục project theo chuẩn nghiên cứu:

  Code LLM Recommend System/
  ├── README.md
  ├── requirements.txt
  ├── .gitignore
  ├── config.py                  ← tập trung tất cả paths
  │
  ├── src/                       ← source code chính
  │   ├── data_prep.py
  │   ├── fine_tune.py
  │   ├── inference_rag.py
  │   └── xai_shap.py
  │
  ├── evaluation/                ← scripts đánh giá
  │   ├── baseline_eval.py
  │   ├── base_rag_eval.py
  │   └── merge_results.py
  │
  ├── notebooks/                 ← Colab notebooks
  │   ├── Wine_Finetune_Colab.ipynb
  │   ├── Wine_Evaluate_Colab.ipynb
  │   ├── generate_eval_notebook.py
  │   └── generate_notebook.py
  │
  ├── data/
  │   ├── raw/
  │   │   └── winemag-data-130k-v2.csv
  │   └── processed/
  │       ├── wine_train_130k.jsonl
  │       ├── wine_val_130k.jsonl
  │       └── wine_test_130k.jsonl
  │
  ├── results/                   ← evaluation outputs
  │   ├── baseline_comparison.csv
  │   ├── final_comparison.csv
  │   ├── bm25_per_query.csv
  │   ├── tfidf_per_query.csv
  │   └── base_rag_results.csv
  │
  ├── models/                    ← model artifacts (.gitignored)
  │   └── lora_wine_model/
  │
  ├── api/static/                ← web interface
  └── chroma_db/                 ← vector store (.gitignored)

Usage:
    python3 reorganize.py          # dry-run (show plan, don't move)
    python3 reorganize.py --apply  # actually move files + update paths
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.resolve()
DRY_RUN = "--apply" not in sys.argv

# ─── Directory structure to create ───────────────────────────────────────────
DIRS = [
    "src",
    "evaluation",
    "notebooks",
    "data/raw",
    "data/processed",
    "results",
    "models",
    "api/static",
]

# ─── File moves: (source, destination) ───────────────────────────────────────
MOVES = [
    # src/
    ("data_prep.py",              "src/data_prep.py"),
    ("fine_tune.py",              "src/fine_tune.py"),
    ("inference_rag.py",          "src/inference_rag.py"),
    ("xai_shap.py",               "src/xai_shap.py"),

    # evaluation/
    ("baseline_eval.py",          "evaluation/baseline_eval.py"),
    ("base_rag_eval.py",          "evaluation/base_rag_eval.py"),
    ("merge_results.py",          "evaluation/merge_results.py"),

    # notebooks/
    ("Wine_Evaluate_Colab.ipynb", "notebooks/Wine_Evaluate_Colab.ipynb"),
    ("Wine_Finetune_Colab.ipynb", "notebooks/Wine_Finetune_Colab.ipynb"),
    ("generate_eval_notebook.py", "notebooks/generate_eval_notebook.py"),
    ("generate_notebook.py",      "notebooks/generate_notebook.py"),

    # data/raw/
    ("winemag-data-130k-v2.csv",  "data/raw/winemag-data-130k-v2.csv"),

    # data/processed/
    ("wine_train_130k.jsonl",      "data/processed/wine_train_130k.jsonl"),
    ("wine_val_130k.jsonl",        "data/processed/wine_val_130k.jsonl"),
    ("wine_test_130k.jsonl",       "data/processed/wine_test_130k.jsonl"),
    ("wine_training_dataset.jsonl","data/processed/wine_training_dataset.jsonl"),
    ("wine_test_dataset.jsonl",    "data/processed/wine_test_dataset.jsonl"),

    # results/
    ("baseline_comparison.csv",   "results/baseline_comparison.csv"),
    ("final_comparison.csv",      "results/final_comparison.csv"),
    ("bm25_per_query.csv",        "results/bm25_per_query.csv"),
    ("tfidf_per_query.csv",       "results/tfidf_per_query.csv"),
    ("base_rag_results.csv",      "results/base_rag_results.csv"),

    # models/
    ("lora_wine_model",           "models/lora_wine_model"),
    ("lora_wine_model.zip",       "models/lora_wine_model.zip"),

    # api/
    ("static",                    "api/static"),
]

# ─── config.py content (generated at root) ────────────────────────────────────
CONFIG_PY = '''\
"""
config.py — Centralized path configuration
==========================================
Import this in any script to get consistent absolute paths.

Usage:
    from config import DATA_RAW, DATA_PROC, RESULTS, MODELS
    df = pd.read_csv(DATA_RAW / "winemag-data-130k-v2.csv")
"""
from pathlib import Path

ROOT       = Path(__file__).parent.resolve()

# Data
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"

# Model artifacts
MODELS     = ROOT / "models"
LORA_MODEL = MODELS / "lora_wine_model"

# Evaluation results
RESULTS    = ROOT / "results"
BASELINE_CSV   = RESULTS / "baseline_comparison.csv"
FINAL_CSV      = RESULTS / "final_comparison.csv"
BM25_CSV       = RESULTS / "bm25_per_query.csv"
TFIDF_CSV      = RESULTS / "tfidf_per_query.csv"
BASE_RAG_CSV   = RESULTS / "base_rag_results.csv"

# Vector store
CHROMA_DB  = ROOT / "chroma_db"

# Web interface
STATIC_DIR = ROOT / "api" / "static"

# Dataset files
WINE_CSV       = DATA_RAW  / "winemag-data-130k-v2.csv"
TRAIN_JSONL    = DATA_PROC / "wine_train_130k.jsonl"
VAL_JSONL      = DATA_PROC / "wine_val_130k.jsonl"
TEST_JSONL     = DATA_PROC / "wine_test_130k.jsonl"

def ensure_dirs():
    """Create all required directories if they don\'t exist."""
    for d in [DATA_RAW, DATA_PROC, MODELS, RESULTS, CHROMA_DB, STATIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
'''

# ─── requirements.txt content ─────────────────────────────────────────────────
REQUIREMENTS = """\
# Core LLM & Training
unsloth
peft>=0.10.0
bitsandbytes>=0.43.0
transformers>=4.40.0
trl<0.9.0
datasets>=2.18.0
accelerate>=0.29.0

# Vector Database
chromadb>=0.4.0

# Web API
fastapi>=0.110.0
uvicorn>=0.28.0

# Evaluation metrics
rouge_score>=0.1.2
bert_score>=0.3.13

# Retrieval baselines
rank_bm25>=0.2.2
scikit-learn>=1.4.0

# XAI
shap>=0.44.0

# Utilities
pandas>=2.2.0
numpy>=1.26.0
tqdm>=4.66.0
"""

# ─── .gitignore content ───────────────────────────────────────────────────────
GITIGNORE = """\
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
*.egg-info/

# Large data files (use DVC or Kaggle API instead)
data/raw/*.csv
data/processed/*.jsonl

# Model artifacts (large binary files)
models/lora_wine_model/
models/*.zip
Archive.zip

# Vector store (rebuilt automatically)
chroma_db/

# Evaluation outputs (regenerated)
results/*.csv

# macOS
.DS_Store

# Notebooks (track only source, not outputs)
notebooks/.ipynb_checkpoints/

# Environment
.env
"""

# ─── __init__.py content ──────────────────────────────────────────────────────
INIT_PY = """\
# Auto-generated by reorganize.py
"""

# ─── Path substitution rules ──────────────────────────────────────────────────
# For each moved Python file, update hardcoded paths to use config.py constants
PATH_PATCHES = {
    "src/data_prep.py": [
        (r'"winemag-data-130k-v2\.csv"',   'str(cfg.WINE_CSV)'),
        (r"'winemag-data-130k-v2\.csv'",   'str(cfg.WINE_CSV)'),
        (r'"wine_train_130k\.jsonl"',      'str(cfg.TRAIN_JSONL)'),
        (r"'wine_train_130k\.jsonl'",      'str(cfg.TRAIN_JSONL)'),
        (r'"wine_val_130k\.jsonl"',        'str(cfg.VAL_JSONL)'),
        (r"'wine_val_130k\.jsonl'",        'str(cfg.VAL_JSONL)'),
        (r'"wine_test_130k\.jsonl"',       'str(cfg.TEST_JSONL)'),
        (r"'wine_test_130k\.jsonl'",       'str(cfg.TEST_JSONL)'),
    ],
    "src/fine_tune.py": [
        (r'"wine_train_130k\.jsonl"',      'str(cfg.TRAIN_JSONL)'),
        (r"'wine_train_130k\.jsonl'",      'str(cfg.TRAIN_JSONL)'),
        (r'"wine_val_130k\.jsonl"',        'str(cfg.VAL_JSONL)'),
        (r"'wine_val_130k\.jsonl'",        'str(cfg.VAL_JSONL)'),
        (r'"lora_wine_model"',             'str(cfg.LORA_MODEL)'),
        (r"'lora_wine_model'",             'str(cfg.LORA_MODEL)'),
    ],
    "src/inference_rag.py": [
        (r'"winemag-data-130k-v2\.csv"',   'str(cfg.WINE_CSV)'),
        (r"'winemag-data-130k-v2\.csv'",   'str(cfg.WINE_CSV)'),
        (r'"lora_wine_model"',             'str(cfg.LORA_MODEL)'),
        (r"'lora_wine_model'",             'str(cfg.LORA_MODEL)'),
        (r'path="\./chroma_db"',           f'path=str(cfg.CHROMA_DB)'),
        (r"path='./chroma_db'",            f"path=str(cfg.CHROMA_DB)"),
        (r'directory="static"',            'directory=str(cfg.STATIC_DIR)'),
        (r"directory='static'",            "directory=str(cfg.STATIC_DIR)"),
    ],
    "src/xai_shap.py": [
        (r'"winemag-data-130k-v2\.csv"',   'str(cfg.WINE_CSV)'),
        (r"'winemag-data-130k-v2\.csv'",   'str(cfg.WINE_CSV)'),
    ],
    "evaluation/baseline_eval.py": [
        (r'"wine_test_130k\.jsonl"',       'str(cfg.TEST_JSONL)'),
        (r"'wine_test_130k\.jsonl'",       'str(cfg.TEST_JSONL)'),
        (r'"winemag-data-130k-v2\.csv"',   'str(cfg.WINE_CSV)'),
        (r"'winemag-data-130k-v2\.csv'",   'str(cfg.WINE_CSV)'),
        (r'"baseline_comparison\.csv"',    'str(cfg.BASELINE_CSV)'),
        (r"'baseline_comparison\.csv'",    'str(cfg.BASELINE_CSV)'),
        (r'"bm25_per_query\.csv"',         'str(cfg.BM25_CSV)'),
        (r"'bm25_per_query\.csv'",         'str(cfg.BM25_CSV)'),
        (r'"tfidf_per_query\.csv"',        'str(cfg.TFIDF_CSV)'),
        (r"'tfidf_per_query\.csv'",        'str(cfg.TFIDF_CSV)'),
    ],
    "evaluation/base_rag_eval.py": [
        (r'"wine_test_130k\.jsonl"',       'str(cfg.TEST_JSONL)'),
        (r"'wine_test_130k\.jsonl'",       'str(cfg.TEST_JSONL)'),
        (r'"baseline_comparison\.csv"',    'str(cfg.BASELINE_CSV)'),
        (r"'baseline_comparison\.csv'",    'str(cfg.BASELINE_CSV)'),
        (r'"base_rag_results\.csv"',       'str(cfg.BASE_RAG_CSV)'),
        (r"'base_rag_results\.csv'",       'str(cfg.BASE_RAG_CSV)'),
    ],
    "evaluation/merge_results.py": [
        (r'"baseline_comparison\.csv"',    'str(cfg.BASELINE_CSV)'),
        (r"'baseline_comparison\.csv'",    'str(cfg.BASELINE_CSV)'),
        (r'"final_comparison\.csv"',       'str(cfg.FINAL_CSV)'),
        (r"'final_comparison\.csv'",       'str(cfg.FINAL_CSV)'),
    ],
}

CONFIG_IMPORT = "import sys, os; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1])); import config as cfg\n"


# ─── Helper functions ─────────────────────────────────────────────────────────

def log(msg, indent=0):
    prefix = "  " * indent
    print(prefix + msg)

def mkdir(path: Path):
    if DRY_RUN:
        log(f"[mkdir] {path.relative_to(ROOT)}")
    else:
        path.mkdir(parents=True, exist_ok=True)

def move(src: Path, dst: Path):
    if not src.exists():
        log(f"[SKIP]  {src.relative_to(ROOT)} (not found)", 1)
        return
    if DRY_RUN:
        log(f"[move]  {src.relative_to(ROOT)}")
        log(f"    →   {dst.relative_to(ROOT)}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        log(f"✅ {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")

def write_file(path: Path, content: str, desc: str):
    if DRY_RUN:
        log(f"[write] {path.relative_to(ROOT)} ({desc})")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log(f"✅ Created {path.relative_to(ROOT)} ({desc})")

def patch_file(rel_path: str, patches: list):
    """Apply regex substitutions to a Python file and prepend config import."""
    fpath = ROOT / rel_path
    if not fpath.exists():
        log(f"[SKIP]  {rel_path} (not found for patching)")
        return
    src = fpath.read_text(encoding="utf-8")
    original = src

    # Add config import after the first docstring or at top
    if "import config as cfg" not in src:
        # Insert after module docstring if present
        if src.startswith('"""') or src.startswith("'''"):
            end = src.index('"""', 3) + 3 if '"""' in src[3:] else src.index("'''", 3) + 3
            src = src[:end] + "\n\n" + CONFIG_IMPORT + src[end:]
        else:
            src = CONFIG_IMPORT + src

    # Apply path patches
    n_patches = 0
    for pattern, replacement in patches:
        new_src, n = re.subn(pattern, replacement, src)
        if n > 0:
            src = new_src
            n_patches += n

    if DRY_RUN:
        log(f"[patch] {rel_path}  ({n_patches} substitutions)")
    else:
        if src != original:
            fpath.write_text(src, encoding="utf-8")
            log(f"✅ Patched {rel_path} ({n_patches} path substitutions)")
        else:
            log(f"  (no changes in {rel_path})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    if DRY_RUN:
        print("  DRY RUN — showing plan (no files moved)")
        print("  Run with --apply to actually reorganize")
    else:
        print("  APPLYING reorganization...")
    print("=" * 62)

    # 1. Create directories
    print("\n── Directories ──────────────────────────────────────────")
    for d in DIRS:
        mkdir(ROOT / d)

    # 2. Generate helper files
    print("\n── Generate files ───────────────────────────────────────")
    write_file(ROOT / "config.py",       CONFIG_PY,    "path constants")
    write_file(ROOT / "requirements.txt", REQUIREMENTS, "pip packages")
    write_file(ROOT / ".gitignore",       GITIGNORE,    "gitignore")
    write_file(ROOT / "src/__init__.py",           INIT_PY, "package init")
    write_file(ROOT / "evaluation/__init__.py",    INIT_PY, "package init")
    write_file(ROOT / "notebooks/__init__.py",     INIT_PY, "package init")

    # 3. Move files
    print("\n── Move files ───────────────────────────────────────────")
    for src_rel, dst_rel in MOVES:
        move(ROOT / src_rel, ROOT / dst_rel)

    # 4. Patch paths in Python files
    print("\n── Patch path references ────────────────────────────────")
    for rel_path, patches in PATH_PATCHES.items():
        patch_file(rel_path, patches)

    # 5. Summary
    print("\n" + "=" * 62)
    if DRY_RUN:
        print("  Dry-run complete. Run with --apply to execute.")
    else:
        print("  ✅ Reorganization complete!")
        print("""
  New structure:
    src/           — core source code
    evaluation/    — baseline + evaluation scripts
    notebooks/     — Colab notebooks + generators
    data/raw/      — original CSV dataset
    data/processed/— train/val/test JSONL
    results/       — evaluation output CSVs
    models/        — lora_wine_model/
    api/static/    — web interface
    chroma_db/     — vector store (unchanged)
    config.py      — centralized path constants
    requirements.txt
    .gitignore
        """)


if __name__ == "__main__":
    main()
