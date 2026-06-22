"""
xai_shap.py — Transparent Scoring Explainer
============================================
SHAP-based feature attribution for a **transparent heuristic scoring
function** used in the Wine Recommendation system.

IMPORTANT — Scientific Honesty Disclaimer
-----------------------------------------
This module does **NOT** explain the LLM's (Llama-3) internal reasoning
or the ChromaDB vector-retrieval ranking.  Instead it explains a separate,
hand-crafted scoring function that combines 5 interpretable features with
fixed weights (price_match 0.30, style_match 0.25, pairing_match 0.20,
region_match 0.15, semantic_sim 0.10).  SHAP KernelExplainer is applied to
this lightweight surrogate scorer so that users can see *which observable
features* favour a given wine, but the attributions should not be
interpreted as an explanation of the language model's generation process.

Usage:
    python3 xai_shap.py              # Run benchmark demo
    from xai_shap import TransparentScoringExplainer
    from xai_shap import explain_recommendation, build_background  # back-compat
"""

import sys, os; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1])); import config as cfg


import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Feature definitions ──────────────────────────────────────────────────────
FEATURE_NAMES = [
    "style_match",    # Query style keywords vs wine variety overlap
    "pairing_match",  # Food pairing compatibility
    "region_match",   # Explicit region/country mention in query
    "semantic_sim",   # Embedding cosine similarity, neutral 0.5 if no embedding
]

FOOD_KEYWORDS = [
    "steak", "seafood", "fish", "salmon", "tuna", "shrimp",
    "cheese", "chicken", "pasta", "dessert", "lamb", "beef",
    "pork", "vegetarian", "vegan", "pizza", "sushi", "oyster",
]

STYLE_KEYWORDS = {
    "light"     : ["pinot", "grigio", "riesling", "sauvignon"],
    "bold"      : ["cabernet", "malbec", "syrah", "shiraz", "zinfandel"],
    "sweet"     : ["moscato", "riesling", "port", "dessert"],
    "dry"       : ["brut", "chardonnay", "sauvignon blanc", "pinot grigio"],
    "sparkling" : ["champagne", "prosecco", "cava", "brut", "sparkling"],
    "red"       : ["cabernet", "merlot", "malbec", "pinot noir", "sangiovese"],
    "white"     : ["chardonnay", "riesling", "sauvignon", "pinot grigio"],
    "rosé"      : ["rose", "rosé", "provence"],
}


# ─── Feature extraction ───────────────────────────────────────────────────────

def price_compatibility(query: str, wine_price) -> float:
    """Score how well query's stated price matches the wine price (0–1)."""
    match = re.search(r'\$\s*(\d+(?:\.\d+)?)', query)
    try:
        wine_price = float(wine_price)
    except (TypeError, ValueError):
        return 0.5
    if not match:
        return 0.5   # no price stated → neutral
    query_price = float(match.group(1))
    if max(query_price, wine_price) == 0:
        return 1.0
    ratio = min(query_price, wine_price) / max(query_price, wine_price)
    return round(float(ratio), 4)


def style_similarity(query: str, variety: str) -> float:
    """
    Match query style adjectives and grape keywords to wine variety.
    Returns 0–1 overlap score.
    """
    q = query.lower()
    v = variety.lower()
    # Direct variety mention
    variety_words = set(v.split())
    query_words   = set(q.split())
    direct_overlap = len(variety_words & query_words) / max(len(variety_words), 1)
    if direct_overlap > 0:
        return min(direct_overlap, 1.0)
    # Style keyword match
    for style, grapes in STYLE_KEYWORDS.items():
        if style in q:
            if any(g in v for g in grapes):
                return 0.8
    return 0.1


def pairing_score(query: str, description: str) -> float:
    """Score food pairing compatibility (0–1)."""
    q = query.lower()
    d = description.lower()
    mentioned = [k for k in FOOD_KEYWORDS if k in q]
    if not mentioned:
        return 0.5   # no food mentioned → neutral
    hits = sum(1 for f in mentioned if f in d)
    return hits / len(mentioned)


def region_match(query: str, country: str) -> float:
    """1.0 if query explicitly mentions the wine's country, else 0.0."""
    if not country:
        return 0.0
    return 1.0 if country.lower() in query.lower() else 0.0


def embedding_cosine_sim(query_emb, wine_emb) -> float:
    """Cosine similarity between two embedding vectors."""
    if query_emb is None or wine_emb is None:
        return 0.5
    a = np.array(query_emb, dtype=float)
    b = np.array(wine_emb,  dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.5


def extract_features(query: str, wine: dict,
                     query_emb=None, wine_emb=None) -> np.ndarray:
    """
    Extract the 4 tabular features from a (query, wine) pair.
    Returns ndarray of shape (4,).
    """
    return np.array([
        style_similarity(query, wine.get("variety", "")),
        pairing_score(query, wine.get("description", "")),
        region_match(query, wine.get("country", "")),
        embedding_cosine_sim(query_emb, wine_emb),
    ], dtype=float)


# ─── Transparent heuristic scoring function (target of SHAP explanation) ─────
#
# These are hand-tuned, FIXED weights — they are NOT learned from data.
# SHAP explains this deterministic linear combination, not the LLM.
_WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15])

_DISCLAIMER = (
    "This explanation covers a transparent heuristic scoring function "
    "(4 hand-crafted features with fixed weights). It does NOT explain "
    "the LLM's internal reasoning or the vector-retrieval ranking."
)


def scoring_fn(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Transparent heuristic relevance scorer.

    Computes a weighted sum of 4 interpretable features.
    This is the function that SHAP explains — it is a simple linear
    surrogate, not the language model.

    Shape: (n_samples, 4) → (n_samples,)
    """
    return feature_matrix @ _WEIGHTS


# ─── Background data builder ──────────────────────────────────────────────────

def build_background(catalog_df: pd.DataFrame,
                     n_samples: int = 100,
                     random_state: int = 42) -> np.ndarray:
    """
    Build background feature matrix for SHAP KernelExplainer.
    Uses a sample of catalog wines with a generic query.
    """
    sample = catalog_df.sample(min(n_samples, len(catalog_df)),
                               random_state=random_state)
    generic_query = "Recommend a good wine"
    bg = np.array([
        extract_features(generic_query, row.to_dict())
        for _, row in sample.iterrows()
    ])
    return bg


# ─── TransparentScoringExplainer ──────────────────────────────────────────────

class TransparentScoringExplainer:
    """
    SHAP-based explainer for the **transparent heuristic scoring function**.

    What this explains
    ------------------
    A deterministic, hand-crafted linear scorer that combines 4 interpretable
    features (style_match, pairing_match, region_match, semantic_sim)
    with fixed weights [0.35, 0.30, 0.20, 0.15].

    What this does NOT explain
    --------------------------
    * The Llama-3 language model's token-level generation process.
    * ChromaDB's embedding-based retrieval ranking.
    * Any learned model parameters — the weights are hand-tuned constants.

    This class computes SHAP values analytically in closed form for a linear model:
    phi_i = w_i * (x_i - E[x_i]), achieving exact attributions with 0ms overhead.
    """

    def __init__(self, background_features: np.ndarray):
        """
        Args:
            background_features: np.ndarray of shape (n_bg, 4) built by
                ``build_background()``.  Serves as the SHAP reference
                distribution.
        """
        self.expected_features = np.mean(background_features, axis=0)
        self.expected_value = float(self.expected_features @ _WEIGHTS)

    # noinspection PyMethodMayBeStatic
    def explain(self, query: str, wine: dict,
                n_shap_samples: int = 100) -> dict:
        """
        Generate SHAP feature attribution for a single (query, wine) pair.

        Args:
            query          : User's natural-language query.
            wine           : Retrieved wine dict (title, price, variety,
                             country, description).
            n_shap_samples : Ignored, as linear SHAP values are computed exactly.

        Returns:
            dict with keys:
                explanation_type   – always ``'transparent_scoring'``
                disclaimer         – scientific-honesty notice
                score_model        – ``'weighted_features_v1'``
                feature_names      – list[str]
                feature_values     – list[float]
                shap_values        – list[float]
                base_value         – float
                explanation_text   – human-readable summary (top-3 features)
                latency_ms         – wall-clock time in milliseconds
        """
        t0       = time.time()
        features = extract_features(query, wine)

        # Exact linear SHAP values: phi_i = w_i * (x_i - E[x_i])
        sv = _WEIGHTS * (features - self.expected_features)

        latency_ms = (time.time() - t0) * 1000

        # Human-readable attribution (top 3 contributing features)
        pairs = sorted(zip(FEATURE_NAMES, sv),
                       key=lambda x: abs(x[1]), reverse=True)
        lines = [f"Wine: {wine.get('title', 'Unknown')}"]
        for feat, val in pairs[:3]:
            arrow = "↑" if val > 0 else "↓"
            lines.append(f"  {arrow} {feat:<16}: {val:+.3f}")
        explanation_text = "\n".join(lines)

        return {
            "explanation_type" : "transparent_scoring",
            "disclaimer"       : _DISCLAIMER,
            "score_model"      : "weighted_features_v1",
            "feature_names"    : FEATURE_NAMES,
            "feature_values"   : features.tolist(),
            "shap_values"      : sv.tolist(),
            "base_value"       : self.expected_value,
            "explanation_text" : explanation_text,
            "latency_ms"       : round(latency_ms, 3),
        }


# ─── Backward-compatible wrapper (used by inference_rag.py) ───────────────────

def explain_recommendation(query: str,
                            wine: dict,
                            background_features: np.ndarray,
                            n_shap_samples: int = 100) -> dict:
    """
    **Deprecated wrapper** — prefer ``TransparentScoringExplainer.explain()``.

    Kept for backward compatibility with ``inference_rag.py`` and evaluation
    scripts.  Creates a one-shot explainer and delegates to
    ``TransparentScoringExplainer.explain()``.
    """
    explainer = TransparentScoringExplainer(background_features)
    return explainer.explain(query, wine, n_shap_samples=n_shap_samples)


# ─── Latency benchmark ────────────────────────────────────────────────────────

def benchmark(n_queries: int = 5):
    """
    Benchmark SHAP attribution latency vs baseline.
    Prints a comparison table.
    """
    import os
    catalog_path = str(cfg.WINE_CSV)
    if not os.path.exists(catalog_path):
        print(f"ERROR: {catalog_path} not found. Run from project root.")
        return

    print("="*60)
    print("  Heuristic SHAP Attribution — Benchmark")
    print("="*60)

    df = pd.read_csv(catalog_path).dropna(
        subset=["country", "variety", "description"]
    ).head(500)
    print(f"Catalog loaded: {len(df):,} wines")

    print("Building background features (100 samples)...")
    t0 = time.time()
    bg = build_background(df, n_samples=100)
    print(f"  Background built in {(time.time()-t0)*1000:.0f}ms")
    print(f"  Shape: {bg.shape}")

    test_queries = [
        "Recommend a bold red Cabernet from US under $50 for steak dinner",
        "Light white wine from France around $30 for seafood",
        "Sweet dessert wine from Italy for cheese pairing",
        "Affordable Malbec from Argentina under $20",
        "Sparkling wine from Spain for a party celebration",
    ]

    print(f"\nRunning {n_queries} SHAP attributions...")
    latencies = []
    for i, q in enumerate(test_queries[:n_queries]):
        wine = df.iloc[i * 10].to_dict()
        result = explain_recommendation(q, wine, bg, n_shap_samples=64)
        latencies.append(result["latency_ms"])
        print(f"\n[Query {i+1}] {q[:55]}...")
        print(f"  Wine   : {wine.get('title','?')[:50]}")
        print(f"  Features: {dict(zip(FEATURE_NAMES, [f'{v:.3f}' for v in result['feature_values']]))}")
        print(f"  SHAP   : {dict(zip(FEATURE_NAMES, [f'{v:+.3f}' for v in result['shap_values']]))}")
        print(f"  Base   : {result['base_value']:.3f}")
        print(f"  Latency: {result['latency_ms']:.0f}ms")

    print("\n" + "="*60)
    print(f"  Avg SHAP latency: {np.mean(latencies):.0f}ms")
    print(f"  Min / Max       : {min(latencies):.0f}ms / {max(latencies):.0f}ms")
    print("  (Base inference w/o SHAP: ~0ms for feature extraction)")
    print("="*60)


if __name__ == "__main__":
    benchmark(n_queries=3)
