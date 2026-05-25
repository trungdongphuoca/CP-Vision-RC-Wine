"""
xai_shap.py
===========
SHAP-based heuristic feature attribution for Wine Recommendation (Tuần 17).

Extracts 5 tabular features from a (query, wine) pair, then applies
SHAP KernelExplainer to explain a transparent heuristic relevance score.
This is not a direct explanation of the LLM or ChromaDB internals.

Ánh xạ: "Xử lý lỗi cho các loại tài liệu đặc thù" → explain why this
wine was chosen over others using feature-level attribution.

Usage:
    python3 xai_shap.py              # Run benchmark demo
    from xai_shap import explain_recommendation, build_background
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
    "price_match",    # How well query budget matches wine price
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
    Extract the 5 tabular features from a (query, wine) pair.
    Returns ndarray of shape (5,).
    """
    return np.array([
        price_compatibility(query, wine.get("price")),
        style_similarity(query, wine.get("variety", "")),
        pairing_score(query, wine.get("description", "")),
        region_match(query, wine.get("country", "")),
        embedding_cosine_sim(query_emb, wine_emb),
    ], dtype=float)


# ─── Heuristic scoring function (used by SHAP) ───────────────────────────────

# Hand-tuned weights. Replace with an actual trained ranker when available.
_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])


def scoring_fn(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Predict a heuristic relevance score for each row in feature_matrix.
    Shape: (n_samples, 5) → (n_samples,)
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


# ─── SHAP explainer ───────────────────────────────────────────────────────────

def explain_recommendation(query: str,
                            wine: dict,
                            background_features: np.ndarray,
                            n_shap_samples: int = 100) -> dict:
    """
    Generate SHAP attribution for a single (query, wine) pair.

    Args:
        query              : User's natural language query
        wine               : Retrieved wine dict (title, price, variety, country, description)
        background_features: np.ndarray (n_bg, 5) from build_background()
        n_shap_samples     : SHAP approximation samples (trade-off speed/accuracy)

    Returns:
        dict with keys: feature_names, feature_values, shap_values,
                        base_value, explanation_text, latency_ms
    """
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")

    t0       = time.time()
    features = extract_features(query, wine).reshape(1, -1)

    explainer   = shap.KernelExplainer(
        scoring_fn, background_features, link="identity"
    )
    shap_values = explainer.shap_values(
        features, nsamples=n_shap_samples, silent=True
    )
    # shap_values shape: (1, 5) for single-output
    if isinstance(shap_values, list):
        sv = np.array(shap_values[0][0])
    else:
        sv = np.array(shap_values[0])

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
        "attribution_type": "heuristic_feature_attribution",
        "score_model": "weighted_features_v1",
        "feature_names"   : FEATURE_NAMES,
        "feature_values"  : features[0].tolist(),
        "shap_values"     : sv.tolist(),
        "base_value"      : float(explainer.expected_value),
        "explanation_text": explanation_text,
        "latency_ms"      : round(latency_ms, 1),
    }


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
        subset=["country", "variety", "description", "price"]
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
