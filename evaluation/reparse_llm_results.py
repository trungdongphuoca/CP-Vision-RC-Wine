"""
reparse_llm_results.py
======================
Re-parse llm_eval_results.csv với parser nâng cao để xử lý:
1. Model output JSON format với key "selected_id" bị truncate
2. Extract Semantic ID từ JSON hoặc text

Usage:
    python3 evaluation/reparse_llm_results.py
    python3 evaluation/reparse_llm_results.py --input results/llm_eval_results.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

import argparse, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input",  default=str(cfg.RESULTS / "llm_eval_results.csv"))
parser.add_argument("--output", default=str(cfg.RESULTS / "llm_eval_results_reparsed.csv"))
args = parser.parse_args()

# ─── Enhanced Semantic ID parser ──────────────────────────────────────────────
ID_PATTERN = re.compile(r'[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-(?:\d{4}|NV)')

def extract_id_from_json_str(text: str):
    """Try to extract Semantic ID from JSON object in text, even if truncated."""
    # Find "selected_id": "XXXX partial or complete
    m = re.search(r'"selected_id"\s*:\s*"([^"]*)"', text)
    if m:
        candidate = m.group(1).strip().upper()
        id_m = ID_PATTERN.search(candidate)
        if id_m:
            return id_m.group(0)
        # partial match — selected_id value may be cut off, try anyway
        if re.match(r'^[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{0,5}', candidate):
            return None  # truncated, can't recover
    return None

def parse_semantic_id_v2(text: str) -> str:
    """
    Enhanced parser — handles:
      1. Standard 'XX-XX-XX-XXXX' anywhere in text
      2. JSON {"selected_id": "XX-XX-XX-XXXX"} (even if truncated)
      3. Bracket format [XX-XX-XX-XXXX]
    """
    text = str(text).strip()

    # Priority 1: direct ID_PATTERN anywhere in text
    m = ID_PATTERN.search(text)
    if m:
        return m.group(0).upper()

    # Priority 2: selected_id in JSON
    result = extract_id_from_json_str(text)
    if result:
        return result

    return "INVALID_ID"

def normalize(s: str) -> str:
    return str(s).strip().upper().replace("[","").replace("]","")

def eval_components(pred_id, target_id):
    p = pred_id.split("-")
    t = normalize(target_id).split("-")
    c = int(p[0] == t[0]) if len(p) > 0 and len(t) > 0 else 0
    v = int(p[2] == t[2]) if len(p) > 2 and len(t) > 2 else 0
    return c, v, int(c and v)

# ─── Load & re-parse ──────────────────────────────────────────────────────────
print("=" * 60)
print("  Re-parse LLM results with enhanced Semantic ID extractor")
print("=" * 60)

df = pd.read_csv(args.input)
print(f"\n  Input     : {args.input}")
print(f"  Samples   : {len(df):,}")
print(f"\n  OLD metrics:")
print(f"    Valid ID Rate  : {df['ValidID'].mean()*100:.2f}%")
print(f"    Exact Match    : {df['ExactMatch'].mean()*100:.2f}%")
print(f"    IntentMatch@1  : {df['IntentMatch@1'].mean()*100:.2f}%")
print(f"    ROUGE-L        : {df['ROUGE_L'].mean():.4f}")

# Re-parse
new_pred_ids    = []
new_valid       = []
new_em          = []
new_country     = []
new_variety     = []
new_intent      = []

improved = 0
for _, row in df.iterrows():
    gen_text   = str(row.get("generated", ""))
    target_id  = str(row.get("target_id", ""))
    old_pred   = str(row.get("pred_id", "INVALID_ID"))

    new_pred = parse_semantic_id_v2(gen_text)
    if new_pred != old_pred and old_pred == "INVALID_ID":
        improved += 1

    target_norm = normalize(target_id)
    is_match    = int(new_pred == target_norm)
    c_m, v_m, int_m = eval_components(new_pred, target_id)

    new_pred_ids.append(new_pred)
    new_valid.append(int(new_pred != "INVALID_ID"))
    new_em.append(is_match)
    new_country.append(c_m)
    new_variety.append(v_m)
    new_intent.append(int_m)

df["pred_id"]       = new_pred_ids
df["ValidID"]       = new_valid
df["ExactMatch"]    = new_em
df["CountryMatch@1"]= new_country
df["VarietyMatch@1"]= new_variety
df["IntentMatch@1"] = new_intent

print(f"\n  Re-parse results (IDs recovered from INVALID: {improved:,}):")
print(f"\n  NEW metrics:")
print(f"    Valid ID Rate  : {df['ValidID'].mean()*100:.2f}%")
print(f"    Exact Match    : {df['ExactMatch'].mean()*100:.2f}%")
print(f"    CountryMatch@1 : {df['CountryMatch@1'].mean()*100:.2f}%")
print(f"    VarietyMatch@1 : {df['VarietyMatch@1'].mean()*100:.2f}%")
print(f"    IntentMatch@1  : {df['IntentMatch@1'].mean()*100:.2f}%")
print(f"    ROUGE-L        : {df['ROUGE_L'].mean():.4f}")
if "BERTScore_F1" in df.columns:
    print(f"    BERTScore F1   : {df['BERTScore_F1'].mean():.4f}")
print(f"    Avg Latency    : {df['latency_ms'].mean():.0f} ms/query")

# ─── Sample correct predictions ───────────────────────────────────────────────
correct = df[df["ExactMatch"] == 1]
if len(correct) > 0:
    print(f"\n  Exact Match examples ({len(correct)} total):")
    for _, r in correct.head(3).iterrows():
        print(f"    ✓ target={r['target_id']}  pred={r['pred_id']}")
else:
    print(f"\n  No exact matches found.")
    print(f"  Top partial matches (IntentMatch):")
    intent_ok = df[df["IntentMatch@1"] == 1]
    for _, r in intent_ok.head(3).iterrows():
        print(f"    ~ target={r['target_id']}  pred={r['pred_id']}")

# ─── Save ─────────────────────────────────────────────────────────────────────
df.to_csv(args.output, index=False)
print(f"\n  ✓ Saved reparsed results → {args.output}")

# ─── Note on truncation ───────────────────────────────────────────────────────
truncated = df[df["pred_id"] == "INVALID_ID"]
print(f"\n  Note: {len(truncated):,} samples still INVALID_ID")
print(f"  These are likely truncated outputs (MAX_NEW_TOKENS=80 was too short).")
print(f"  Recommendation: re-run Colab eval with MAX_NEW_TOKENS=150")
print()
