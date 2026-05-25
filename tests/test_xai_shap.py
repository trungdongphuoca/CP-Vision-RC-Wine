import numpy as np

from src.xai_shap import (
    FEATURE_NAMES,
    extract_features,
    price_compatibility,
    scoring_fn,
    style_similarity,
)


def test_price_compatibility_scores_stated_budget_ratio():
    assert price_compatibility("under $50", 25) == 0.5
    assert price_compatibility("no budget mentioned", 25) == 0.5


def test_style_similarity_matches_style_keyword_to_variety():
    assert style_similarity("bold red for steak", "Cabernet Sauvignon") == 0.8
    assert style_similarity("pinot noir please", "Pinot Noir") == 1.0


def test_extract_features_has_expected_shape_and_neutral_embedding():
    features = extract_features(
        "Recommend a bold red from US under $50",
        {
            "price": 40,
            "variety": "Cabernet Sauvignon",
            "country": "US",
            "description": "Dark fruit and spice.",
        },
    )

    assert features.shape == (len(FEATURE_NAMES),)
    assert features[-1] == 0.5


def test_scoring_fn_returns_one_score_per_row():
    matrix = np.ones((3, len(FEATURE_NAMES)))

    scores = scoring_fn(matrix)

    assert scores.shape == (3,)
