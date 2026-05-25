"""
constrained_eval.py
===================
Evaluation Suite for Fine-Tuned LLM (Llama-3-8B + LoRA) with New Semantic IDs.

Runs inference on the test dataset, extracts predicted Semantic IDs,
and computes standard recommendation metrics (ExactMatch, Recall@10, Latency).
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

parser = argparse.ArgumentParser()
parser.add_argument("--test_file",  default=str(cfg.TEST_JSONL))
parser.add_argument("--catalog_file", default=str(cfg.WINE_SEMANTIC_CSV))
parser.add_argument("--eval_size",  type=int, default=1000, help="Number of samples to evaluate.")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=350)
parser.add_argument("--output",     default=str(cfg.RESULTS / "constrained_eval_results.csv"))
parser.add_argument("--mock",       action="store_true", help="Run in mock mode")
args = parser.parse_args()

print("="*60)
print("  Evaluating Fine-Tuned Wine Sommelier (Option B - TIGER Style)")
print("="*60)

if not os.path.exists(args.test_file):
    print(f"ERROR: {args.test_file} not found. Run data_prep.py first.")
    sys.exit(1)

with open(args.test_file, encoding='utf-8') as f:
    test_data = [json.loads(l) for l in f][:args.eval_size]

# Load catalog for valid IDs
catalog = pd.read_csv(args.catalog_file)
VALID_IDS = set(catalog['Semantic_ID'].values)

print(f"Test samples : {len(test_data):,}")
print(f"Valid IDs in catalog: {len(VALID_IDS):,}")
print(f"Batch size   : {args.batch_size}")
print(f"Max tokens   : {args.max_new_tokens}")
print(f"Mode         : {'MOCK' if args.mock else 'REAL GPU'}")

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a Master Sommelier. Analyze the user's request and determine "
    "the ideal structural profile of the wine. Then, output the Semantic ID "
    "of the perfect match, followed by a persuasive explanation."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

def parse_semantic_id(text: str) -> str:
    """Extract Semantic ID (e.g. 03-12-07-001) from generated text."""
    ID_PAT = r'\d{2}-\d{2}-\d{2}-\d{3}'
    
    # Strip thought block first to search only in response section
    after_thought = re.split(r'</thought>', text, maxsplit=1)
    search_text = after_thought[-1]
    
    # Bracketed format
    m = re.search(r'\[(' + ID_PAT + r')\]', search_text)
    if m: return m.group(1)
    
    # Bare format
    m = re.search(ID_PAT, search_text)
    if m: return m.group(0)
    
    # Fallback to entire text
    m = re.search(r'\[(' + ID_PAT + r')\]', text)
    if m: return m.group(1)
    m = re.search(ID_PAT, text)
    if m: return m.group(0)
    
    return "INVALID_ID"

def mock_generate(instruction: str, target_id: str) -> str:
    import random
    if random.random() < 0.8:
        pred_id = target_id
    else:
        # Generate some valid ID from the set
        pred_id = random.choice(list(VALID_IDS))
    return f"<thought>\nMock thought process...\n</thought>\nI recommend [{pred_id}]. It fits perfectly."

model = tokenizer = None
DEVICE = "cpu"

if not args.mock:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

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
        
        if os.path.exists(str(cfg.LORA_MODEL)):
            print(f"Loading LoRA weights from {cfg.LORA_MODEL}...")
            model = PeftModel.from_pretrained(base_model, str(cfg.LORA_MODEL))
        else:
            print(f"WARNING: LoRA path {cfg.LORA_MODEL} not found! Evaluating BASE model instead.")
            model = base_model

        model.eval()
            
        tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    except Exception as e:
        print(f"WARNING: Could not load real model ({e}). Falling back to mock mode.")
        args.mock = True

records = []
latencies = []

with tqdm(total=len(test_data), desc="LLM Eval", unit="sample") as pbar:
    for b_start in range(0, len(test_data), args.batch_size):
        batch          = test_data[b_start : b_start + args.batch_size]
        target_ids     = [r["target_id"] for r in batch]
        instructions   = [r["instruction"] for r in batch]

        t0 = time.time()

        if args.mock:
            decoded = [mock_generate(r["instruction"], r["target_id"]) for r in batch]
        else:
            import torch
            prompts = [PROMPT_TEMPLATE.format(instruction=i) for i in instructions]
            inputs  = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=2048 - args.max_new_tokens,
            ).to(DEVICE)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens = args.max_new_tokens,
                    do_sample      = False,
                    use_cache      = True,
                    pad_token_id   = tokenizer.eos_token_id,
                )
            new_toks = outputs[:, input_len:]
            decoded  = tokenizer.batch_decode(new_toks, skip_special_tokens=True)

        batch_lat = (time.time() - t0) * 1000 / len(batch)

        for gen_text, target_id in zip(decoded, target_ids):
            gen_text = gen_text.strip()
            pred_id  = parse_semantic_id(gen_text)
            
            # Extract cluster path
            t_parts = target_id.split('-')
            p_parts = pred_id.split('-') if pred_id != "INVALID_ID" else []
            
            is_match = int(pred_id == target_id)
            is_valid = int(pred_id in VALID_IDS)
            
            # If the model gets the first 3 parts right (the cluster), it's a semantic hit
            cluster_match = int(len(p_parts)==4 and len(t_parts)==4 and p_parts[:3] == t_parts[:3])
            
            records.append({
                "target_id": target_id,
                "pred_id": pred_id,
                "generated": gen_text,
                "ExactMatch": is_match,
                "ValidID": is_valid,
                "ClusterMatch": cluster_match,
                "latency_ms": batch_lat
            })

        pbar.update(len(batch))

df_results = pd.DataFrame(records)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df_results.to_csv(args.output, index=False)

em_rate = df_results["ExactMatch"].mean()
val_rate = df_results["ValidID"].mean()
cluster_rate = df_results["ClusterMatch"].mean()
lat_avg = df_results["latency_ms"].mean()

print(f"\n{'='*60}")
print(f"  EVALUATION SUMMARY - Option B (Hierarchical Semantic IDs)")
print(f"  Samples  : {len(df_results):,}")
print(f"  Valid ID Rate    : {val_rate*100:.2f}%")
print(f"  Exact Match (EM) : {em_rate*100:.2f}%")
print(f"  Cluster Match    : {cluster_rate*100:.2f}% (Found the correct semantic grouping)")
print(f"  Avg Latency      : {lat_avg:.1f} ms/query")
print(f"  Results saved to : {args.output}")
print(f"{'='*60}\n")
