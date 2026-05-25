"""
generate_eval_notebook.py
=========================
Generates Wine_Evaluate_Colab.ipynb — full evaluation notebook for the
fine-tuned LLM (Llama-3-8B + LoRA) Generative Retrieval system.

FIX: Beam search REMOVED — Unsloth uses a tuple KV-cache that is
     incompatible with transformers ≥4.41 beam-search `reorder_cache()`.
     We use greedy decoding (do_sample=False, num_beams=1) which is
     fully compatible and matches standard GR-paper evaluation protocol.

Metrics:
  Retrieval : Exact Match (EM = Recall@1 for greedy LLM)
  Generation: ROUGE-L, BERTScore F1
  Baselines : Recall@K, NDCG@K, MRR from baseline_eval.py (local run)

Dataset:
  Kaggle Wine Reviews (winemag-data-130k-v2.csv)
  Aeberhard, S. & Forina, M. (2017). Wine Reviews. Kaggle.
  https://www.kaggle.com/datasets/zynicide/wine-reviews
"""

import json

# ─────────────────────────────────────────────────────────────────────────────

CELL_INSTALL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "%%capture\n"
        "!pip install unsloth\n"
        "!pip install --no-deps xformers 'trl<0.9.0' peft accelerate bitsandbytes\n"
        "!pip install datasets rouge_score bert_score nltk\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "print('Packages installed.')"
    )
}

CELL_UPLOAD = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "from google.colab import files\n"
        "\n"
        "print('1. Upload wine_test_130k.jsonl:')\n"
        "_ = files.upload()\n"
        "\n"
        "print('2. Upload baseline_comparison.csv (from local baseline_eval.py):')\n"
        "_ = files.upload()\n"
        "\n"
        "print('3. Upload lora_wine_model.zip:')\n"
        "_ = files.upload()\n"
        "!unzip -o lora_wine_model.zip -d . && echo 'Model unzipped.'\n"
    )
}

CELL_LOAD_MODEL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ================================================================\n"
        "# FIX: transformers 5.5.0 + Unsloth 2026.5.x on T4 (14.5 GB)\n"
        "# Error  : ValueError: Some modules dispatched on CPU/disk\n"
        "# Cause  : transformers >=5.x changed device_map auto-dispatch;\n"
        "#          when VRAM is marginally tight it offloads layers to CPU,\n"
        "#          but BitsAndBytes 4-bit does NOT support CPU offload.\n"
        "# Fix    : (1) gc + empty_cache before load to maximise free VRAM\n"
        "#          (2) device_map='cuda:0'  — pin ALL layers to GPU\n"
        "#          (3) HF-transformers fallback if Unsloth still fails\n"
        "# ================================================================\n"
        "import gc, torch, time\n"
        "from unsloth import FastLanguageModel\n"
        "from peft import PeftModel\n"
        "\n"
        "# ── Step 0: Maximise free VRAM before loading ─────────────────────\n"
        "gc.collect()\n"
        "torch.cuda.empty_cache()\n"
        "torch.cuda.synchronize()\n"
        "free_gb  = (torch.cuda.get_device_properties(0).total_memory\n"
        "            - torch.cuda.memory_allocated()) / 1e9\n"
        "total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
        "print(f'GPU: {torch.cuda.get_device_name(0)}  |  "
        "Free VRAM: {free_gb:.2f} / {total_gb:.2f} GB')\n"
        "\n"
        "MAX_SEQ_LEN = 2048\n"
        "model = tokenizer = None\n"
        "\n"
        "# ── Step 1: Unsloth fast path (preferred) ─────────────────────────\n"
        "try:\n"
        "    model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "        model_name    = 'unsloth/llama-3-8b-bnb-4bit',\n"
        "        max_seq_length= MAX_SEQ_LEN,\n"
        "        dtype         = None,      # auto: bf16 or fp16\n"
        "        load_in_4bit  = True,\n"
        "        device_map    = 'cuda:0',  # KEY FIX: pin all layers to GPU\n"
        "    )\n"
        "    print('✓ Unsloth fast path: base model loaded.')\n"
        "except Exception as e_unsloth:\n"
        "    print(f'⚠ Unsloth failed: {e_unsloth}')\n"
        "    print('→ Falling back to standard HuggingFace transformers...')\n"
        "\n"
        "    # ── Step 1b: HF-transformers fallback ─────────────────────────\n"
        "    from transformers import (\n"
        "        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n"
        "    )\n"
        "    gc.collect(); torch.cuda.empty_cache()\n"
        "    bnb_cfg = BitsAndBytesConfig(\n"
        "        load_in_4bit              = True,\n"
        "        bnb_4bit_quant_type       = 'nf4',\n"
        "        bnb_4bit_compute_dtype    = torch.float16,\n"
        "        bnb_4bit_use_double_quant = True,\n"
        "    )\n"
        "    base_model_id = 'unsloth/llama-3-8b-bnb-4bit'\n"
        "    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)\n"
        "    base = AutoModelForCausalLM.from_pretrained(\n"
        "        base_model_id,\n"
        "        quantization_config = bnb_cfg,\n"
        "        device_map          = 'cuda:0',  # force GPU only\n"
        "        torch_dtype         = torch.float16,\n"
        "    )\n"
        "    model = PeftModel.from_pretrained(base, 'lora_wine_model')\n"
        "    model.eval()\n"
        "    print('✓ HF-transformers fallback: model + LoRA loaded.')\n"
        "\n"
        "# ── Step 2: Load LoRA adapter (Unsloth path) ──────────────────────\n"
        "if model is not None and not isinstance(model, PeftModel):\n"
        "    try:\n"
        "        model = PeftModel.from_pretrained(model, 'lora_wine_model')\n"
        "        print('✓ LoRA adapter loaded: lora_wine_model/')\n"
        "    except Exception as e_lora:\n"
        "        print(f'WARNING: LoRA load failed ({e_lora}).')\n"
        "        print('  Running base model only (ablation / no fine-tuning).')\n"
        "\n"
        "# ── Step 3: Inference mode ────────────────────────────────────────\n"
        "try:\n"
        "    FastLanguageModel.for_inference(model)\n"
        "    print('✓ Unsloth for_inference() enabled (2-4x speedup).')\n"
        "except Exception:\n"
        "    model.eval()\n"
        "    print('✓ Standard model.eval() mode.')\n"
        "\n"
        "# ── Step 4: Tokenizer padding for batched decoding ────────────────\n"
        "tokenizer.pad_token    = tokenizer.eos_token\n"
        "tokenizer.padding_side = 'left'   # CRITICAL for decoder-only batches\n"
        "\n"
        "# ── Summary ───────────────────────────────────────────────────────\n"
        "alloc_gb = torch.cuda.memory_allocated() / 1e9\n"
        "print(f'\\nVRAM after load : {alloc_gb:.2f} GB used')\n"
        "print(f'VRAM headroom   : {total_gb - alloc_gb:.2f} GB free (for activations)')\n"
        "print(f'LoRA adapter    : {isinstance(model, PeftModel)}')\n"
        "print('\\n✓ Ready for evaluation!')\n"
    )
}

CELL_LOAD_DATA = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "import json, re, os, math, time\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from rouge_score import rouge_scorer\n"
        "from tqdm import tqdm\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "\n"
        "def load_jsonl(path):\n"
        "    with open(path) as f:\n"
        "        return [json.loads(l) for l in f]\n"
        "\n"
        "test_data = load_jsonl('wine_test_130k.jsonl')\n"
        "print(f'Loaded {len(test_data):,} test samples.')\n"
        "print('Example:', test_data[0]['instruction'], '->', test_data[0]['target_id'])\n"
    )
}

CELL_CONFIG = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ================================================================\n"
        "#  CONFIGURATION\n"
        "#  Dataset: Kaggle Wine Reviews (winemag-data-130k-v2.csv)\n"
        "#  Cite   : https://www.kaggle.com/datasets/zynicide/wine-reviews\n"
        "#\n"
        "#  IMPORTANT — Beam search is DISABLED.\n"
        "#  Reason  : Unsloth stores KV-cache as a plain Python tuple.\n"
        "#            Transformers >= 4.41 beam-search calls\n"
        "#            past_key_values.reorder_cache(beam_idx) which fails\n"
        "#            with AttributeError: 'tuple' has no 'reorder_cache'.\n"
        "#  Fix     : Use greedy decoding (num_beams=1, do_sample=False).\n"
        "#            This is the standard eval protocol in GR papers\n"
        "#            (DSI, NCI) when reporting Exact Match / Recall@1.\n"
        "# ================================================================\n"
        "# ================================================================\n"
        "BATCH_SIZE     = 8       # T4-safe default (increase to 12/16 if stable)\n"
        "MAX_NEW_TOKENS = 80      # sufficient for Semantic ID + brief note\n"
        "EVAL_SIZE      = 1000    # Match baseline_eval.py\n"
        "SAVE_EVERY     = 200     # Save more frequently since eval size is smaller\n"
        "RESUME_FROM    = 0\n"
        "CHECKPOINT     = 'eval_checkpoint.csv'\n"
        "FINAL_CSV      = 'llm_eval_results.csv'\n"
        "SUMMARY_CSV    = 'evaluation_results_summary.csv'\n"
        "BERT_MODE      = 'fast'  # 'skip' | 'fast' | 'full'\n"
        "BERT_FAST_N    = 300     # used when BERT_MODE='fast'\n"
        "\n"
        "PROMPT_TEMPLATE = (\n"
        "    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n'\n"
        "    'You are a Master Sommelier. Analyze the user\\'s request and '\n"
        "    'determine the ideal structural profile of the wine. Then, output '\n"
        "    'the Semantic ID of the perfect match, followed by a persuasive '\n"
        "    'explanation.'\n"
        "    '<|eot_id|><|start_header_id|>user<|end_header_id|>\\n'\n"
        "    '{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n'\n"
        ")\n"
        "\n"
        "# Slice test data to match baseline evaluation size\n"
        "test_data = test_data[:EVAL_SIZE]\n"
        "n = len(test_data)\n"
        "est = n / BATCH_SIZE * 1.8 / 60\n"
        "print(f'Config: batch={BATCH_SIZE}, greedy, max_new_tokens={MAX_NEW_TOKENS}, bert_mode={BERT_MODE}')\n"
        "print(f'Samples to eval: {n:,}  |  Est. generation ~{est:.1f} min on T4 GPU')\n"
    )
}

CELL_METRICS_FN = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ── Helper metric functions ──────────────────────────────────────\n"
        "def normalize_semantic_id(s):\n"
        "    return str(s).strip().upper().replace('[', '').replace(']', '')\n"
        "\n"
        "def parse_semantic_id(text):\n"
        "    \"\"\"Extract Semantic ID (e.g. US-NAPA-CABE-2015) from generated text.\"\"\"\n"
        "    m = re.search(r'[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-[A-Z0-9]{2,5}-(?:\\d{4}|NV)', text)\n"
        "    return normalize_semantic_id(m.group(0)) if m else 'INVALID_ID'\n"
        "\n"
        "def extract_explanation_text(gen_text):\n"
        "    # Drop optional thought tags and keep outward-facing explanation only\n"
        "    body = gen_text.split('</thought>')[-1].strip()\n"
        "    if body.startswith('<thought>'):\n"
        "        body = body.replace('<thought>', '').strip()\n"
        "    return body\n"
        "\n"
        "def eval_components(pred_id, target_id):\n"
        "    p_parts = pred_id.split('-')\n"
        "    t_parts = normalize_semantic_id(target_id).split('-')\n"
        "    country_match = int(p_parts[0] == t_parts[0]) if len(p_parts) > 0 and len(t_parts) > 0 else 0\n"
        "    variety_match = int(p_parts[2] == t_parts[2]) if len(p_parts) > 2 and len(t_parts) > 2 else 0\n"
        "    intent_match = int(country_match and variety_match)\n"
        "    return country_match, variety_match, intent_match\n"
        "\n"
        "rouge_sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)\n"
        "print('Metric helpers ready.')\n"
    )
}

CELL_MAIN_LOOP = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ================================================================\n"
        "#  MAIN EVALUATION LOOP\n"
        "#  Strategy : Batched GREEDY decoding (do_sample=False, num_beams=1)\n"
        "#  Compatible: Unsloth tuple KV-cache — no reorder_cache() called\n"
        "#  Primary metric: Exact Match (EM) = LLM Recall@1\n"
        "#  Also computed : ROUGE-L on explanation text\n"
        "# ================================================================\n"
        "\n"
        "records, start_idx = [], 0\n"
        "\n"
        "if RESUME_FROM > 0 and os.path.exists(CHECKPOINT):\n"
        "    df_ck     = pd.read_csv(CHECKPOINT)\n"
        "    records   = df_ck.to_dict('records')\n"
        "    start_idx = len(records)\n"
        "    print(f'Resumed from {start_idx:,} samples.')\n"
        "else:\n"
        "    print('Starting fresh evaluation.')\n"
        "\n"
        "eval_data = test_data[start_idx:]\n"
        "\n"
        "with tqdm(total=len(eval_data), desc='LLM Eval', unit='sample') as pbar:\n"
        "    for b_start in range(0, len(eval_data), BATCH_SIZE):\n"
        "        batch          = eval_data[b_start : b_start + BATCH_SIZE]\n"
        "        prompts        = [PROMPT_TEMPLATE.format(instruction=r['instruction']) for r in batch]\n"
        "        target_ids     = [r['target_id'] for r in batch]\n"
        "        expected_resps = [r['response']  for r in batch]\n"
        "\n"
        "        # ── Tokenize (left-padded for decoder-only batch) ────────\n"
        "        inputs = tokenizer(\n"
        "            prompts,\n"
        "            return_tensors='pt',\n"
        "            padding=True,\n"
        "            truncation=True,\n"
        "            max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,\n"
        "        ).to('cuda')\n"
        "        input_len = inputs['input_ids'].shape[1]\n"
        "\n"
        "        # ── GREEDY generation ────────────────────────────────────\n"
        "        # do_sample=False + num_beams=1 (default) → greedy\n"
        "        # use_cache=True  → Unsloth tuple cache, OK for greedy\n"
        "        # temperature=None, top_p=None required when do_sample=False\n"
        "        t0 = time.time()\n"
        "        try:\n"
        "            with torch.inference_mode():\n"
        "                outputs = model.generate(\n"
        "                    **inputs,\n"
        "                    max_new_tokens = MAX_NEW_TOKENS,\n"
        "                    do_sample      = False,\n"
        "                    temperature    = None,\n"
        "                    top_p          = None,\n"
        "                    use_cache      = True,\n"
        "                    pad_token_id   = tokenizer.eos_token_id,\n"
        "                )\n"
        "        except torch.cuda.OutOfMemoryError:\n"
        "            torch.cuda.empty_cache()\n"
        "            # Retry with smaller micro-batch by splitting current batch in two\n"
        "            mid = max(1, len(batch)//2)\n"
        "            left_inputs = {k: v[:mid] for k, v in inputs.items()}\n"
        "            right_inputs = {k: v[mid:] for k, v in inputs.items()}\n"
        "            outs = []\n"
        "            for part in (left_inputs, right_inputs):\n"
        "                with torch.inference_mode():\n"
        "                    outs.append(model.generate(\n"
        "                        **part,\n"
        "                        max_new_tokens=MAX_NEW_TOKENS,\n"
        "                        do_sample=False,\n"
        "                        temperature=None,\n"
        "                        top_p=None,\n"
        "                        use_cache=True,\n"
        "                        pad_token_id=tokenizer.eos_token_id,\n"
        "                    ))\n"
        "            outputs = torch.cat(outs, dim=0)\n"
        "        batch_lat = (time.time() - t0) * 1000 / len(batch)\n"
        "\n"
        "        # ── Decode NEW tokens only ───────────────────────────────\n"
        "        new_toks = outputs[:, input_len:]\n"
        "        decoded  = tokenizer.batch_decode(new_toks, skip_special_tokens=True)\n"
        "\n"
        "        # ── Per-sample metrics ───────────────────────────────────\n"
        "        for gen_text, target_id, expected in zip(decoded, target_ids, expected_resps):\n"
        "            gen_text = gen_text.strip()\n"
        "            pred_id  = parse_semantic_id(gen_text)\n"
        "            target_norm = normalize_semantic_id(target_id)\n"
        "            is_match = int(pred_id == target_norm)\n"
        "            c_m, v_m, int_m = eval_components(pred_id, target_id)\n"
        "\n"
        "            gen_resp = extract_explanation_text(gen_text)\n"
        "            rouge_l  = rouge_sc.score(expected, gen_resp)['rougeL'].fmeasure\n"
        "\n"
        "            records.append({\n"
        "                'target_id'  : target_id,\n"
        "                'target_id_norm': target_norm,\n"
        "                'pred_id'    : pred_id,\n"
        "                'generated'  : gen_text[:300],\n"
        "                'generated_explanation': gen_resp[:600],\n"
        "                'expected_response': expected[:600],\n"
        "                'ValidID'    : int(pred_id != 'INVALID_ID'),\n"
        "                'ExactMatch' : is_match,\n"
        "                'CountryMatch@1': c_m,\n"
        "                'VarietyMatch@1': v_m,\n"
        "                'IntentMatch@1' : int_m,\n"
        "                'ROUGE_L'    : rouge_l,\n"
        "                'latency_ms' : batch_lat,\n"
        "            })\n"
        "\n"
        "        pbar.update(len(batch))\n"
        "\n"
        "        # ── Incremental checkpoint ───────────────────────────────\n"
        "        done = start_idx + b_start + len(batch)\n"
        "        if done % SAVE_EVERY < BATCH_SIZE:\n"
        "            pd.DataFrame(records).to_csv(CHECKPOINT, index=False)\n"
        "            em = np.mean([r['ExactMatch'] for r in records])\n"
        "            rl = np.mean([r['ROUGE_L']    for r in records])\n"
        "            pbar.set_postfix({'EM': f'{em:.3f}', 'ROUGE-L': f'{rl:.3f}', 'n': done})\n"
        "\n"
        "# ── Final save ───────────────────────────────────────────────────\n"
        "llm_df = pd.DataFrame(records)\n"
        "llm_df.to_csv(FINAL_CSV, index=False)\n"
        "print(f'\\n=== LLM Evaluation Results ===')\n"
        "print(f'Samples       : {len(llm_df):,}')\n"
        "print(f'Exact Match   : {llm_df[\"ExactMatch\"].mean()*100:.2f}%')\n"
        "print(f'Intent Match  : {llm_df[\"IntentMatch@1\"].mean()*100:.2f}%')\n"
        "print(f'Valid ID rate : {llm_df[\"ValidID\"].mean()*100:.2f}%')\n"
        "print(f'ROUGE-L       : {llm_df[\"ROUGE_L\"].mean():.4f}')\n"
        "print(f'Avg Latency   : {llm_df[\"latency_ms\"].mean():.1f} ms/query')\n"
        "print(f'Saved to      : {FINAL_CSV}')\n"
    )
}

CELL_BERTSCORE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ── BERTScore (run after main loop to avoid GPU OOM) ─────────────\n"
        "from bert_score import score as bert_score\n"
        "\n"
        "llm_df = pd.read_csv(FINAL_CSV)\n"
        "if BERT_MODE == 'skip':\n"
        "    print('BERTScore skipped by config (BERT_MODE=skip).')\n"
        "else:\n"
        "    if BERT_MODE == 'fast':\n"
        "        llm_eval = llm_df.head(min(BERT_FAST_N, len(llm_df))).copy()\n"
        "        print(f'BERTScore fast mode on {len(llm_eval):,} samples.')\n"
        "    else:\n"
        "        llm_eval = llm_df.copy()\n"
        "        print(f'BERTScore full mode on {len(llm_eval):,} samples.')\n"
        "\n"
        "    hypotheses = llm_eval['generated_explanation'].fillna('').tolist()\n"
        "    references = llm_eval['expected_response'].fillna('').tolist()\n"
        "\n"
        "    BERT_BATCH = 32\n"
        "    all_f1     = []\n"
        "    n_batches  = math.ceil(len(hypotheses) / BERT_BATCH)\n"
        "    print(f'BERTScore over {len(hypotheses):,} samples ({n_batches} batches)...')\n"
        "\n"
        "    for b in tqdm(range(n_batches), desc='BERTScore'):\n"
        "        hyp_b = hypotheses[b*BERT_BATCH : (b+1)*BERT_BATCH]\n"
        "        ref_b = references[b*BERT_BATCH : (b+1)*BERT_BATCH]\n"
        "        _, _, F1 = bert_score(hyp_b, ref_b, lang='en', verbose=False, device='cuda')\n"
        "        all_f1.extend(F1.tolist())\n"
        "\n"
        "    llm_eval['BERTScore_F1'] = all_f1\n"
        "    mean_f1 = float(np.mean(all_f1))\n"
        "\n"
        "    if BERT_MODE == 'full':\n"
        "        llm_df['BERTScore_F1'] = llm_eval['BERTScore_F1']\n"
        "    else:\n"
        "        llm_df['BERTScore_F1'] = float('nan')\n"
        "        llm_df.loc[:len(llm_eval)-1, 'BERTScore_F1'] = llm_eval['BERTScore_F1'].values\n"
        "\n"
        "    llm_df.to_csv(FINAL_CSV, index=False)\n"
        "    print(f'BERTScore F1 (mean on evaluated subset): {mean_f1:.4f}')\n"
    )
}

CELL_SUMMARY = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ================================================================\n"
        "#  FULL COMPARISON TABLE\n"
        "#  Merges LLM results + baseline_comparison.csv from local run\n"
        "# ================================================================\n"
        "\n"
        "llm_df = pd.read_csv(FINAL_CSV)\n"
        "\n"
        "# LLM: greedy → single prediction → EM == Recall@1\n"
        "em_val = llm_df['ExactMatch'].mean()\n"
        "llm_summary = {\n"
        "    'Recall@1'      : em_val,\n"
        "    'ExactMatch(EM)': em_val,\n"
        "    'IntentMatch@1' : llm_df['IntentMatch@1'].mean() if 'IntentMatch@1' in llm_df.columns else 0.0,\n"
        "    'CountryMatch@1': llm_df['CountryMatch@1'].mean() if 'CountryMatch@1' in llm_df.columns else 0.0,\n"
        "    'VarietyMatch@1': llm_df['VarietyMatch@1'].mean() if 'VarietyMatch@1' in llm_df.columns else 0.0,\n"
        "    'ValidIDRate'   : llm_df['ValidID'].mean() if 'ValidID' in llm_df.columns else float('nan'),\n"
        "    'ROUGE_L'       : llm_df['ROUGE_L'].mean(),\n"
        "    'BERTScore_F1'  : llm_df['BERTScore_F1'].mean() if 'BERTScore_F1' in llm_df.columns else float('nan'),\n"
        "    'Latency_ms'    : llm_df['latency_ms'].mean(),\n"
        "}\n"
        "\n"
        "# Load baselines\n"
        "summaries = {}\n"
        "if os.path.exists('baseline_comparison.csv'):\n"
        "    base_df = pd.read_csv('baseline_comparison.csv', index_col=0)\n"
        "    for m in base_df.index:\n"
        "        summaries[m] = base_df.loc[m].to_dict()\n"
        "    print(f'Baselines loaded: {list(base_df.index)}')\n"
        "else:\n"
        "    print('WARNING: baseline_comparison.csv not found. Upload it.')\n"
        "\n"
        "summaries['LLM-LoRA (Proposed)'] = llm_summary\n"
        "\n"
        "result_df = pd.DataFrame(summaries).T\n"
        "result_df.index.name = 'Method'\n"
        "result_df.to_csv(SUMMARY_CSV)\n"
        "\n"
        "display_cols = ['Recall@1','Recall@5','Recall@10','NDCG@5','MRR',\n"
        "                'ExactMatch(EM)','CountryMatch@1','VarietyMatch@1','IntentMatch@1','IntentMatch@10','ValidIDRate',\n"
        "                'ROUGE_L','BERTScore_F1','Latency_ms']\n"
        "display_cols = [c for c in display_cols if c in result_df.columns]\n"
        "\n"
        "print('\\n' + '='*90)\n"
        "print('  EVALUATION RESULTS — Kaggle Wine Reviews')\n"
        "print('  https://www.kaggle.com/datasets/zynicide/wine-reviews')\n"
        "print('  LLM: greedy (Recall@1=EM) | Baselines: top-K retrieval')\n"
        "print('='*90)\n"
        "print(result_df[display_cols].to_string(\n"
        "    float_format=lambda x: f'{x:.4f}' if not pd.isna(x) else '   —  '\n"
        "))\n"
        "print('='*90)\n"
        "\n"
        "bm25_r10 = summaries.get('BM25', {}).get('Recall@10', 'N/A')\n"
        "try:    bm25_r10_str = f'{float(bm25_r10):.4f}'\n"
        "except: bm25_r10_str = str(bm25_r10)\n"
        "print(f'Key: LLM EM={em_val*100:.2f}% (single-shot) vs BM25 Recall@10={bm25_r10_str}')\n"
        "print(f'Saved: {SUMMARY_CSV}')\n"
        "\n"
        "from google.colab import files\n"
        "files.download(SUMMARY_CSV)\n"
        "files.download(FINAL_CSV)\n"
        "print('Files downloaded.')\n"
    )
}

CELL_LATEX = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# ── Generate LaTeX table ─────────────────────────────────────────\n"
        "result_df = pd.read_csv(SUMMARY_CSV, index_col=0)\n"
        "latex_cols = ['Recall@1','Recall@5','NDCG@5','IntentMatch@1','IntentMatch@10','ValidIDRate','MRR','ROUGE_L','BERTScore_F1']\n"
        "latex_cols = [c for c in latex_cols if c in result_df.columns]\n"
        "\n"
        "print('\\\\begin{table}[ht]')\n"
        "print('\\\\centering')\n"
        "print('\\\\caption{Comparison on Kaggle Wine Reviews~\\\\cite{wine_kaggle_2017}}')\n"
        "print('\\\\label{tab:main_results}')\n"
        "print('\\\\begin{tabular}{l' + 'r'*len(latex_cols) + '}')\n"
        "print('\\\\hline')\n"
        "header = ' & '.join(['\\\\textbf{Method}'] + [f'\\\\textbf{{{c}}}' for c in latex_cols])\n"
        "print(header + ' \\\\\\\\')\n"
        "print('\\\\hline')\n"
        "for method, row in result_df[latex_cols].iterrows():\n"
        "    is_prop = 'LLM' in str(method) or 'Proposed' in str(method)\n"
        "    m_str   = f'\\\\textbf{{{method}}}' if is_prop else method\n"
        "    vals    = ' & '.join(f'{v:.4f}' if pd.notna(v) else '--' for v in row)\n"
        "    print(f'{m_str} & {vals} \\\\\\\\')\n"
        "print('\\\\hline')\n"
        "print('\\\\end{tabular}')\n"
        "print('\\\\end{table}')\n"
        "\n"
        "print('\\n% BibTeX:')\n"
        "print('@misc{wine_kaggle_2017,')\n"
        "print('  author={Aeberhard, Stefan and Forina, M.},')\n"
        "print('  title={{Wine Reviews}}, year={2017}, publisher={Kaggle},')\n"
        "print('  url={https://www.kaggle.com/datasets/zynicide/wine-reviews}}')\n"
    )
}

# ─── Assemble & write notebook ────────────────────────────────────────────────
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            "# Đánh giá Mô hình Generative Recommendation — Llama-3-8B + LoRA\n\n"
            "**Dataset:** Kaggle Wine Reviews (`winemag-data-130k-v2.csv`)  \n"
            "**Citation:** Aeberhard, S. (2017). Wine Reviews. Kaggle.  \n"
            "**URL:** https://www.kaggle.com/datasets/zynicide/wine-reviews\n\n"
            "## Evaluation Protocol\n"
            "| Component | Detail |\n"
            "|-----------|--------|\n"
            "| Decoding  | **Greedy** (`do_sample=False`, `num_beams=1`) — compatible with Unsloth |\n"
            "| LLM metric | **Exact Match (EM)** = Recall@1 (single-shot Semantic ID prediction) |\n"
            "| Text quality | **ROUGE-L**, **BERTScore F1** on explanation text |\n"
            "| Baselines | BM25, TF-IDF CF — Recall@K, NDCG@K, MRR (from `baseline_eval.py`) |\n\n"
            "> **Note on Beam Search:** Disabled due to Unsloth KV-cache incompatibility  \n"
            "> (`AttributeError: 'tuple' has no 'reorder_cache'`). Greedy EM is the  \n"
            "> standard metric in Generative Retrieval papers (DSI, NCI, GENRE)."
        )
    },
    CELL_INSTALL,
    CELL_UPLOAD,
    CELL_LOAD_MODEL,
    CELL_LOAD_DATA,
    CELL_CONFIG,
    CELL_METRICS_FN,
    CELL_MAIN_LOOP,
    CELL_BERTSCORE,
    CELL_SUMMARY,
    CELL_LATEX,
]

# Convert string source to list-of-lines for proper .ipynb format
def to_lines(src):
    if isinstance(src, str):
        lines = src.splitlines(keepends=True)
        return lines
    return src  # already a list

for cell in cells:
    if "source" in cell:
        cell["source"] = to_lines(cell["source"])

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("Wine_Evaluate_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print("Wine_Evaluate_Colab.ipynb regenerated.")
print("  Decoding : Greedy (beam search removed — Unsloth tuple-cache fix)")
print("  Metrics  : EM, ROUGE-L, BERTScore")
print("  Baselines: Recall@K, NDCG@K, MRR loaded from baseline_comparison.csv")
