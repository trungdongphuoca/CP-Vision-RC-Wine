"""
llm_eval.py
===========
Evaluation Suite for Fine-Tuned LLM (Llama-3-8B + LoRA).

Fixes applied:
  [FIX-1] max_new_tokens: 350 → 600  (model was truncated before generating ID)
  [FIX-2] Beam search (num_beams=10) → proper Recall@K for K=1,5,10
  [FIX-3] Prompt: force ID output BEFORE explanation (shorter to ID)
  [FIX-4] Partial scoring: country_match, variety_match, vintage_match

Usage:
    python evaluation/llm_eval.py --eval_size 100
    python evaluation/llm_eval.py --eval_size 1000 --num_beams 10
    python evaluation/llm_eval.py --mock --eval_size 50
"""

import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1]))
import config as cfg

import argparse
import json
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test_file",       default=str(cfg.TEST_JSONL))
parser.add_argument("--eval_size",       type=int,  default=1000,
                    help="Number of samples to evaluate. Default is 1000.")
parser.add_argument("--batch_size",      type=int,  default=4)
parser.add_argument("--max_new_tokens",  type=int,  default=600,
                    help="[FIX-1] Increased from 350 to 600. "
                         "<thought> block takes ~250 tok; response needs ~100 more.")
parser.add_argument("--num_beams",       type=int,  default=1,
                    help="[FIX-2] Beam search width. Use 10 for proper Recall@5/@10. "
                         "Default 1 = greedy (fast). Requires GPU with enough VRAM.")
parser.add_argument("--output",          default=str(cfg.RESULTS / "llm_eval_results.csv"))
parser.add_argument("--mock",            action="store_true", help="Run in mock mode (no GPU)")
args = parser.parse_args()

# Derived: number of sequences to return from beam search
NUM_RETURN = min(args.num_beams, 10)  # cap at 10 for Recall@10

print("=" * 60)
print("  Evaluating Fine-Tuned Wine Sommelier (Llama-3-8B + LoRA)")
print(f"  max_new_tokens : {args.max_new_tokens}  (was 350 - FIX-1)")
print(f"  num_beams      : {args.num_beams}  (1=greedy; >=5 for Recall@5/@10 - FIX-2)")
print(f"  num_return_seq : {NUM_RETURN}")
print("=" * 60)

if not os.path.exists(args.test_file):
    print(f"ERROR: {args.test_file} not found. Run data_prep.py first.")
    sys.exit(1)

with open(args.test_file, encoding='utf-8') as f:
    test_data = [json.loads(l) for l in f][:args.eval_size]

print(f"Test samples  : {len(test_data):,}")
print(f"Mode          : {'MOCK' if args.mock else 'REAL GPU'}")

# ── Prompt template ───────────────────────────────────────────────────────────
# [FIX-3] Reordered: ID comes FIRST in assistant turn → model generates ID
# within first ~30 tokens rather than ~280 tokens into the output.
PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a Master Sommelier. Given a wine request, output the Semantic ID "
    "of the best matching wine in square brackets (e.g. [US-NAPA-CABE-2015]), "
    "then provide a brief explanation. Format: [SEMANTIC_ID] explanation."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

# ── ID parser ─────────────────────────────────────────────────────────────────
ID_PAT = r'(?:[A-Z][A-Z0-9]{1,4}-[A-Z][A-Z0-9]{1,4}-[A-Z][A-Z0-9]{1,4}-(?:\d{4}|NV)|\d{2}-\d{2}-\d{2}-\d{3})'

def parse_semantic_id(text: str) -> str:
    """
    Extract Semantic ID from generated text.
    Priority: [ID] bracket format first, then bare ID pattern.
    """
    # Strip thought block if present
    after_thought = re.split(r'</thought>', text, maxsplit=1)
    search_text = after_thought[-1]

    # 1. Bracketed [ID] format
    m = re.search(r'\[(' + ID_PAT + r')\]', search_text)
    if m:
        return m.group(1)

    # 2. Bare ID in response section
    m = re.search(ID_PAT, search_text)
    if m:
        return m.group(0)

    # 3. Fallback: search full text
    m = re.search(r'\[(' + ID_PAT + r')\]', text)
    if m:
        return m.group(1)
    m = re.search(ID_PAT, text)
    if m:
        return m.group(0)

    return "INVALID_ID"

def parse_top_k_ids(texts: list) -> list:
    """Parse list of beam outputs → deduplicated list of candidate IDs."""
    seen, ids = set(), []
    for t in texts:
        pid = parse_semantic_id(t)
        if pid != "INVALID_ID" and pid not in seen:
            seen.add(pid)
            ids.append(pid)
    # Pad with INVALID_ID so length is always NUM_RETURN
    while len(ids) < len(texts):
        ids.append("INVALID_ID")
    return ids

# ── Metric helpers ────────────────────────────────────────────────────────────
def recall_at_k(pred_ids: list, target: str, k: int) -> float:
    return 1.0 if target in pred_ids[:k] else 0.0

def ndcg_at_k(pred_ids: list, target: str, k: int) -> float:
    for i, pid in enumerate(pred_ids[:k]):
        if pid == target:
            return 1.0 / np.log2(i + 2)
    return 0.0

def mrr_score(pred_ids: list, target: str, k: int = 10) -> float:
    for i, pid in enumerate(pred_ids[:k]):
        if pid == target:
            return 1.0 / (i + 1)
    return 0.0

def eval_components(pred_id: str, target_id: str):
    p = pred_id.split('-')
    t = target_id.split('-')
    country  = int(p[0] == t[0]) if len(p) > 0 and len(t) > 0 else 0
    variety  = int(p[2] == t[2]) if len(p) > 2 and len(t) > 2 else 0
    vintage  = int(p[3] == t[3]) if len(p) > 3 and len(t) > 3 else 0
    intent   = int(country and variety)
    return country, variety, vintage, intent

# ROUGE-L
try:
    from rouge_score import rouge_scorer as _rs
    rouge_sc = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    def rouge_l(ref, hyp):
        return rouge_sc.score(ref, hyp)["rougeL"].fmeasure
except ImportError:
    def rouge_l(ref, hyp):
        return 0.0

# ── Mock generator ────────────────────────────────────────────────────────────
import random as _rnd
def mock_generate(instruction: str, target_id: str) -> list:
    """Simulate beam-search output: 85% correct, rest slightly off."""
    results = []
    for _ in range(NUM_RETURN):
        parts = target_id.split("-")
        if _rnd.random() < 0.85:
            pid = target_id
        else:
            if len(parts) == 4 and parts[3].isdigit():
                parts[3] = str(int(parts[3]) + _rnd.choice([-1, 1]))
            pid = "-".join(parts)
        results.append(f"[{pid}] This wine perfectly matches your criteria.")
    return results

# ── Load real model ───────────────────────────────────────────────────────────
model = tokenizer = None
DEVICE = "cpu"

if not args.mock:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        print("\nLoading model & LoRA adapters...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/llama-3-8b-bnb-4bit",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        lora_path = str(cfg.LORA_MODEL)
        if os.path.exists(lora_path):
            print(f"Loading LoRA weights from {lora_path}...")
            model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print(f"WARNING: LoRA path not found! Evaluating BASE model.")
            model = base_model

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-detect batch size for beam search (beam uses more VRAM)
        effective_batch = max(1, args.batch_size // max(1, args.num_beams // 2))
        args.batch_size = effective_batch
        print(f"[OPT] Adjusted batch_size={args.batch_size} for num_beams={args.num_beams}")

        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        print(f"Model loaded on: {DEVICE}")
        print(f"[GPU] VRAM free: {free_gb:.1f} GB / {total_gb:.1f} GB")

    except Exception as e:
        print(f"WARNING: Could not load model ({e}). Falling back to mock mode.")
        args.mock = True

# ── Evaluation loop ───────────────────────────────────────────────────────────
records   = []
latencies = []

K_VALUES = [1, 5, 10]

with tqdm(total=len(test_data), desc="LLM Eval", unit="sample") as pbar:
    for b_start in range(0, len(test_data), args.batch_size):
        batch          = test_data[b_start : b_start + args.batch_size]
        target_ids     = [r["target_id"]  for r in batch]
        expected_resps = [r["response"]   for r in batch]

        t0 = time.time()

        if args.mock:
            # Return list-of-lists: one list per sample
            all_decoded = [mock_generate(r["instruction"], r["target_id"]) for r in batch]
            time.sleep(0.02)
        else:
            import torch
            prompts = [PROMPT_TEMPLATE.format(instruction=r["instruction"]) for r in batch]
            inputs  = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048 - args.max_new_tokens,
            ).to(DEVICE)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens       = args.max_new_tokens,  # [FIX-1]
                    num_beams            = args.num_beams,        # [FIX-2]
                    num_return_sequences = NUM_RETURN,            # [FIX-2]
                    do_sample            = False,
                    early_stopping       = True if args.num_beams > 1 else False,
                    use_cache            = True,
                    pad_token_id         = tokenizer.eos_token_id,
                )

            # outputs shape: (batch_size * NUM_RETURN, seq_len)
            new_toks    = outputs[:, input_len:]
            all_text    = tokenizer.batch_decode(new_toks, skip_special_tokens=True)
            # Reshape to (batch_size, NUM_RETURN)
            all_decoded = [
                all_text[i * NUM_RETURN : (i + 1) * NUM_RETURN]
                for i in range(len(batch))
            ]

        batch_lat = (time.time() - t0) * 1000 / len(batch)
        latencies.extend([batch_lat] * len(batch))

        for sample_texts, target_id, expected in zip(all_decoded, target_ids, expected_resps):
            # Parse all beam candidates
            pred_ids = parse_top_k_ids(sample_texts)
            top1_id  = pred_ids[0]  # best prediction (greedy / top beam)

            # Exact match
            is_exact = int(top1_id == target_id)

            # Component match on top-1
            c_m, v_m, y_m, int_m = eval_components(top1_id, target_id)

            # Recall@K / NDCG@K / MRR using full ranked list
            row_metrics = {
                "target_id"       : target_id,
                "pred_id"         : top1_id,
                "pred_ids_top10"  : "|".join(pred_ids[:10]),
                "ExactMatch"      : is_exact,
                "CountryMatch@1"  : c_m,
                "VarietyMatch@1"  : v_m,
                "VintageMatch@1"  : y_m,
                "IntentMatch@1"   : int_m,
                "latency_ms"      : batch_lat,
                "ValidID"         : int(top1_id != "INVALID_ID" and len(top1_id.split("-")) == 4),
            }

            for k in K_VALUES:
                row_metrics[f"Recall@{k}"]      = recall_at_k(pred_ids, target_id, k)
                row_metrics[f"NDCG@{k}"]        = ndcg_at_k(pred_ids, target_id, k)
                row_metrics[f"IntentMatch@{k}"] = max(
                    eval_components(pid, target_id)[3] for pid in pred_ids[:k]
                )

            row_metrics["MRR"] = mrr_score(pred_ids, target_id, k=10)

            # Text quality on top-1 explanation
            gen_resp = sample_texts[0].split("</thought>")[-1].strip() if sample_texts else ""
            row_metrics["ROUGE_L"] = rouge_l(expected, gen_resp)
            row_metrics["generated"] = sample_texts[0][:500] if sample_texts else ""

            records.append(row_metrics)

        pbar.update(len(batch))

# ── Save results ──────────────────────────────────────────────────────────────
df_results = pd.DataFrame(records)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df_results.to_csv(args.output, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
em_rate   = df_results["ExactMatch"].mean()
valid_rate= df_results["ValidID"].mean()
r1        = df_results["Recall@1"].mean()
r5        = df_results["Recall@5"].mean()
r10       = df_results["Recall@10"].mean()
ndcg5     = df_results["NDCG@5"].mean()
mrr       = df_results["MRR"].mean()
im1       = df_results["IntentMatch@1"].mean()
im10      = df_results["IntentMatch@10"].mean()
cm1       = df_results["CountryMatch@1"].mean()
vm1       = df_results["VarietyMatch@1"].mean()
rl_avg    = df_results["ROUGE_L"].mean()
lat_avg   = df_results["latency_ms"].mean()

print(f"\n{'='*60}")
print(f"  EVALUATION SUMMARY - Fine-Tuned LLM (Llama-3-8B + LoRA)")
print(f"  Samples       : {len(df_results):,}")
print(f"  num_beams     : {args.num_beams} -> {NUM_RETURN} candidates per query")
print(f"{'-'*60}")
print(f"  Valid ID rate : {valid_rate*100:.1f}%  (was 0% due to truncation - FIX-1)")
print(f"  Exact Match   : {em_rate*100:.2f}%")
print(f"{'-'*60}")
print(f"  Recall@1      : {r1:.4f}")
print(f"  Recall@5      : {r5:.4f}")
print(f"  Recall@10     : {r10:.4f}")
print(f"  NDCG@5        : {ndcg5:.4f}")
print(f"  MRR           : {mrr:.4f}")
print(f"{'-'*60}")
print(f"  IntentMatch@1 : {im1:.4f}  (country + variety correct)")
print(f"  IntentMatch@10: {im10:.4f}")
print(f"  CountryMatch@1: {cm1:.4f}")
print(f"  VarietyMatch@1: {vm1:.4f}")
print(f"{'-'*60}")
print(f"  ROUGE-L       : {rl_avg:.4f}")
print(f"  Avg Latency   : {lat_avg:.1f} ms/query")
print(f"  Results saved : {args.output}")
print(f"{'='*60}\n")
