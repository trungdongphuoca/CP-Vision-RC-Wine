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
WINE_SEMANTIC_CSV = DATA_PROC / "wine_catalog_semantic.csv"
TRAIN_JSONL    = DATA_PROC / "wine_train_130k.jsonl"
VAL_JSONL      = DATA_PROC / "wine_val_130k.jsonl"
TEST_JSONL     = DATA_PROC / "wine_test_130k.jsonl"

def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_RAW, DATA_PROC, MODELS, RESULTS, CHROMA_DB, STATIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def get_device() -> str:
    """Check if CUDA is available and fully compatible (including embedding kernels)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        # Test if embedding works on GPU (Blackwell compatibility test)
        emb = torch.nn.Embedding(2, 2).cuda()
        x = torch.tensor([0]).cuda()
        emb(x)
        return "cuda"
    except Exception:
        return "cpu"
