"""
attention_xai.py — LLM Attention-Based Explainability
=====================================================
Extracts attention weights from the last transformer layer of a causal LM
(e.g. Llama-3-8B) to show which input tokens the model focused on when
generating a recommendation.

Unlike the heuristic SHAP explainer in ``xai_shap.py`` (which explains a
hand-crafted scoring function), this module provides a **genuine glimpse
into the LLM's internal processing** by inspecting its self-attention maps.

Limitations
-----------
* Attention ≠ full explanation — attention weights show token-level focus
  but do not capture the causal contribution of each token to the output.
* Only the **last layer's mean-head attention** is reported; earlier layers
  may carry complementary information.
* Memory-intensive for long prompts on GPU.

Usage::

    from attention_xai import extract_attention_highlights
    highlights = extract_attention_highlights(model, tokenizer, prompt)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch


def extract_attention_highlights(
    model,
    tokenizer,
    prompt: str,
    *,
    top_k: int = 10,
    device: Optional[str] = None,
) -> Dict:
    """
    Run a forward pass with ``output_attentions=True`` and return the
    top-k tokens by mean attention weight in the last layer.

    Parameters
    ----------
    model : transformers.PreTrainedModel | peft.PeftModel
        The loaded causal-LM (may be a PeftModel wrapping a base model).
    tokenizer : transformers.PreTrainedTokenizerBase
        Matching tokenizer.
    prompt : str
        The full input prompt whose tokens will be scored.
    top_k : int, default 10
        Number of highest-attention tokens to return.
    device : str | None
        Override device (e.g. ``"cuda"``).  If *None*, inferred from
        ``model.device`` or falls back to CPU.

    Returns
    -------
    dict
        ``attention_highlights`` : list[{token: str, attention: float}]
            Sorted descending by attention weight.
        ``total_tokens``         : int
        ``latency_ms``           : float
        ``layer_index``          : int   (which layer was used)
        ``disclaimer``           : str

    Raises
    ------
    RuntimeError
        If model or tokenizer is None.
    """
    if model is None or tokenizer is None:
        return {
            "attention_highlights": [],
            "total_tokens": 0,
            "latency_ms": 0.0,
            "layer_index": -1,
            "disclaimer": "Model not loaded — attention extraction skipped.",
        }

    t0 = time.time()

    # ── Resolve device ────────────────────────────────────────────────────
    if device is None:
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "cpu"

    # ── Tokenize ──────────────────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── Forward pass (no grad, output attentions) ─────────────────────────
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                use_cache=False,
            )
    except Exception as e:
        return {
            "attention_highlights": [],
            "total_tokens": 0,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "layer_index": -1,
            "disclaimer": f"Attention extraction failed: {e}",
        }

    # ── Extract last-layer attention ──────────────────────────────────────
    # outputs.attentions is a tuple of (batch, n_heads, seq_len, seq_len)
    attentions = outputs.attentions
    if attentions is None or len(attentions) == 0:
        return {
            "attention_highlights": [],
            "total_tokens": 0,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "layer_index": -1,
            "disclaimer": "Model did not return attention weights.",
        }

    last_layer_attn = attentions[-1]          # (1, n_heads, seq_len, seq_len)
    layer_index = len(attentions) - 1

    # Mean across heads → (seq_len, seq_len)
    mean_attn = last_layer_attn[0].mean(dim=0)   # (seq_len, seq_len)

    # Attention *to* each token position, averaged across all query positions
    # This gives a single importance score per input token.
    token_importance = mean_attn.mean(dim=0).cpu().float().numpy()  # (seq_len,)

    # ── Decode tokens and build result ────────────────────────────────────
    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]

    # Pair tokens with scores, filter out special / whitespace-only tokens
    scored = []
    for tok_str, score in zip(tokens, token_importance):
        clean = tok_str.strip()
        if not clean or clean in {"<s>", "</s>", "<pad>", "<unk>",
                                   "<|begin_of_text|>", "<|end_of_text|>",
                                   "<|eot_id|>"}:
            continue
        scored.append({"token": tok_str, "attention": round(float(score), 6)})

    # Sort descending and keep top_k
    scored.sort(key=lambda x: x["attention"], reverse=True)
    highlights = scored[:top_k]

    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "attention_highlights": highlights,
        "total_tokens": len(input_ids),
        "latency_ms": latency_ms,
        "layer_index": layer_index,
        "disclaimer": (
            "Attention weights show token-level focus in the last transformer "
            "layer (averaged across heads). They indicate where the model "
            "looked but are not a complete causal explanation of its output."
        ),
    }
