import pandas as pd
import numpy as np

from src.data_prep import (
    create_prompt,
    split_by_semantic_id,
)

def clean_text(t):
    if not t or pd.isna(t):
        return "UNKN"
    cleaned = "".join(c for c in str(t).upper() if c.isalnum())
    return cleaned[:4]

def add_semantic_ids(df):
    df = df.copy()
    df["Semantic_ID"] = df.apply(
        lambda r: f"{clean_text(r['country'])}-{clean_text(r.get('province',''))}-{clean_text(r['variety'])}-{int(r.get('price', 0))}",
        axis=1
    )
    return df

def assert_no_group_overlap(train_df, val_df, test_df):
    train_ids = set(train_df["Semantic_ID"])
    val_ids = set(val_df["Semantic_ID"])
    test_ids = set(test_df["Semantic_ID"])
    assert not (train_ids & val_ids), "Overlap between train and val"
    assert not (train_ids & test_ids), "Overlap between train and test"
    assert not (val_ids & test_ids), "Overlap between val and test"

def sample_catalog():
    rows = []
    for idx in range(12):
        rows.append({
            "country": "US" if idx < 6 else "France",
            "province": "California" if idx < 6 else "Bordeaux",
            "variety": "Cabernet Sauvignon" if idx % 2 == 0 else "Merlot",
            "title": f"Producer {idx // 2} Estate {2010 + idx // 2}",
            "description": f"Sample tasting note {idx}",
            "price": float(20 + idx),
        })
    return pd.DataFrame(rows)


def test_clean_text_returns_four_character_token():
    assert clean_text("New York") == "NEWY"
    assert clean_text(None) == "UNKN"


def test_group_split_has_no_semantic_id_overlap():
    df = add_semantic_ids(sample_catalog())

    train_df, val_df, test_df = split_by_semantic_id(
        df,
        train_ratio=0.6,
        val_ratio=0.2,
        random_state=7,
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert_no_group_overlap(train_df, val_df, test_df)


def test_create_prompt_handles_missing_price_without_nan_text():
    row = add_semantic_ids(sample_catalog()).iloc[0].copy()
    row["price"] = float("nan")

    prompt = create_prompt(row)

    assert "$nan" not in prompt["instruction"]
    assert "flexible budget" in prompt["instruction"]
