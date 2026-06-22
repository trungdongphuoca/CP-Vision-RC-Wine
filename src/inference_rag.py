import sys, os
_p = __import__('pathlib').Path(__file__).resolve()
sys.path.insert(0, str(_p.parents[1]))
sys.path.insert(0, str(_p.parent))
import config as cfg
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import chromadb
import os
import time
import numpy as np
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ── CV / Label Scanner module ─────────────────────────────────────────────────
try:
    from cv_wine import (
        WineLabelScanner,
        find_in_catalog,
        find_similar_wines,
        build_tasting_notes,
        get_aroma_profile,
        get_food_pairings,
    )
    wine_scanner   = WineLabelScanner()
    CV_AVAILABLE   = True
except ImportError as _cv_err:
    wine_scanner   = None
    CV_AVAILABLE   = False
    print(f"[WARN] cv_wine not available: {_cv_err}")

# ── XAI module — Transparent Scoring Explainer (heuristic SHAP) ────────────
try:
    from xai_shap import (
        explain_recommendation,
        build_background,
        extract_features,
        FEATURE_NAMES,
        TransparentScoringExplainer,
    )
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("[WARN] xai_shap not found — XAI disabled. Run: pip install shap")

# ── GNN Retrieval module ─────────────────────────────────────────────────────
_gnn_import_ok = False


# ── Attention-based LLM Explainability ───────────────────────────────────
try:
    from attention_xai import extract_attention_highlights
    ATTENTION_XAI_AVAILABLE = True
except ImportError:
    ATTENTION_XAI_AVAILABLE = False
    print("[WARN] attention_xai not found — attention visualization disabled.")

app = FastAPI(title="Wine Recommendation API")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB))
collection = chroma_client.get_or_create_collection(name="wine_inventory")

# Global Variables
tokenizer         = None
model             = None
shap_background   = None   # np.ndarray (n_bg, 5) for SHAP KernelExplainer
catalog_df_cache  = None   # cached DataFrame for SHAP background + CV catalog lookup
gnn_retriever     = None   # GNNRetriever instance for hybrid retrieval
scoring_explainer = None   # Global TransparentScoringExplainer instance

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: List[Message] = Field(default_factory=list)

app.mount("/static", StaticFiles(directory=str(cfg.STATIC_DIR)), name="static")

@app.get("/")
def read_root():
    return FileResponse(cfg.STATIC_DIR / "index.html")

@app.on_event("startup")
def load_models_and_data():
    global tokenizer, model, shap_background, catalog_df_cache, gnn_retriever, scoring_explainer, collection

    # ── 1. Populate Vector DB ─────────────────────────────────────────────────
    print("Initializing Vector DB Inventory...")
    try:
        # Recreate collection to ensure X-Wines is loaded and no old entries exist
        try:
            chroma_client.delete_collection(name="wine_inventory")
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(name="wine_inventory")
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, str(cfg.WINE_CSV))
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["country","variety","description","title"]).head(5000)
        docs, metadatas, ids = [], [], []
        for idx, row in df.iterrows():
            docs.append(row["description"])
            metadatas.append({
                "title"  : row["title"],
                "country": row["country"],
                "variety": row["variety"],
            })
            ids.append(str(idx))
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
        print(f"Loaded {len(ids)} wines into ChromaDB.")
    except Exception as e:
        print(f"[WARN] DB init skipped: {e}")

    # ── 2. Load Fine-Tuned LLM (Llama-3-8B + LoRA) ───────────────────────────
    if not torch.cuda.is_available():
        print("[INFO] CUDA is not available. Skipping LLM loading, running in Mock LLM mode (local mode).")
    else:
        print("Loading LLM (Llama-3-8B + LoRA)...")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                "unsloth/llama-3-8b-bnb-4bit",
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model     = PeftModel.from_pretrained(base_model, str(cfg.LORA_MODEL))
            tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit")
            print("LLM loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load LLM: {e}")

    # ── 3. Build SHAP background + load catalog for CV search ─────────────────
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                str(cfg.WINE_CSV))
        catalog_df_cache = pd.read_csv(csv_path).dropna(
            subset=["country", "variety", "description", "title"]
        )
        print(f"[INFO] Catalog loaded: {len(catalog_df_cache):,} wines")
    except Exception as e:
        print(f"[WARN] Catalog load failed: {e}")

    if XAI_AVAILABLE and catalog_df_cache is not None:
        try:
            shap_background = build_background(catalog_df_cache, n_samples=100)
            scoring_explainer = TransparentScoringExplainer(shap_background)
            print(f"SHAP background built and explainer initialized: {shap_background.shape}")
        except Exception as e:
            print(f"[WARN] SHAP background build failed or explainer init failed: {e}")
    else:
        print("[INFO] XAI disabled (xai_shap module not available).")

    # ── 4. Load GNN retriever for hybrid retrieval ────────────────────────────
    if _gnn_import_ok:
        try:
            gnn_retriever = GNNRetriever()
            if gnn_retriever.is_available:
                print("[GNN] Hybrid retrieval enabled — GNN embeddings loaded.")
            else:
                print("[GNN] Embeddings not found, using ChromaDB-only retrieval")
                gnn_retriever = None
        except Exception as e:
            print(f"[GNN] Embeddings not found, using ChromaDB-only retrieval: {e}")
            gnn_retriever = None
    else:
        print("[GNN] Embeddings not found, using ChromaDB-only retrieval")

    # ── 5. Pre-load OCR scanner ────────────────────────────────────────────────
    if CV_AVAILABLE and wine_scanner is not None:
        try:
            wine_scanner.load(load_clip=False)   # CLIP loaded on demand
            print("[CV] Wine label scanner ready.")
        except Exception as e:
            print(f"[WARN] Scanner pre-load failed: {e}")

def generate_profile(query: str, history: List[Message] = []) -> str:
    """Uses the fine-tuned LLM to generate the ideal wine profile/Semantic ID."""
    if model is None or tokenizer is None:
        print("Warning: LLM not loaded. Using Mock Semantic ID.")
        
        # In mock mode, we use the entire conversation history to extract keywords
        full_text = " ".join([m.content for m in history]) + " " + query
        q = full_text.lower()
        if "ital" in q: return "ITAL-TUSC-SANG-2015"
        if "franc" in q: return "FRAN-BORD-REDB-2015"
        if "argentin" in q: return "ARGE-MEND-MALB-2015"
        if "port" in q: return "PORT-DOUR-PORT-2015"
        if "spain" in q or "spanish" in q: return "SPAI-RIOJ-TEMP-2015"
        return "US-CALI-CABE-2013" # Default mock semantic ID

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a Master Sommelier. Analyze the user's request and output ONLY the ideal wine profile descriptors "
        "(e.g., variety, country, dryness, body, tannins) as search keywords. "
        "Do NOT write any conversational text, explanation, or thought process. Max 10 words.<|eot_id|>"
    )
    
    for msg in history:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>"
        
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    # Pre-populate thought block to bypass LLM thinking time
    prompt += "<thought> I will extract search keywords for the user query. </thought>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=25, use_cache=True)
        
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    import re as _re
    # Strip <thought>...</thought> sections that Llama sometimes emits
    clean = _re.sub(r"<thought>.*?</thought>", "", generated_text, flags=_re.DOTALL)
    clean = _re.sub(r"<think>.*?</think>",    "", clean,          flags=_re.DOTALL)
    # Remove raw JSON lines leaking from CoT
    lines = [l for l in clean.splitlines() if not l.strip().startswith("{")]
    return " ".join(lines).strip() or generated_text


def _clean_llm(text: str) -> str:
    """Strip chain-of-thought artifacts and dialog repetition from LLM outputs."""
    import re as _re
    # Truncate at unfinished thought blocks
    if "<thought>" in text and "</thought>" not in text:
        text = text.split("<thought>")[0]
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>")[0]
        
    # Remove completed thought/think blocks
    text = _re.sub(r"<thought>.*?</thought>", "", text, flags=_re.DOTALL)
    text = _re.sub(r"<think>.*?</think>",    "", text, flags=_re.DOTALL)
    
    # Truncate at template markers
    stop_markers = [
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
        r"<\|eot_id\|>",
        r"<\|begin_of_text\|>",
        r"<\|end_of_text\|>",
    ]
    for marker in stop_markers:
        matches = list(_re.finditer(marker, text))
        if matches:
            text = text[:matches[0].start()]
            
    # Truncate at prompt-repetition and hallucinated dialogue markers
    repetition_patterns = [
        r"userıldığında",
        r"\buser\b",
        r"\bassistant\b",
        r"\bsystem\b",
        r"\buser\s*:",
        r"\bassistant\s*:",
    ]
    for rep in repetition_patterns:
        pattern = _re.compile(rep, _re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if matches:
            text = text[:matches[0].start()]

    # Truncate at dialog headers (e.g. "user:" or "assistant:" at the beginning of a line, or a line containing just "user" or "assistant")
    header_pattern = _re.compile(r"^\s*(?:user|assistant)\s*(?::|$)", _re.IGNORECASE | _re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    if matches:
        text = text[:matches[0].start()]
        
    # Clean whitespace and lines
    lines = []
    for line in text.splitlines():
        line_strip = line.strip()
        # skip empty lines and JSON-like leaking lines
        if not line_strip or line_strip.startswith("{"):
            continue
        lines.append(line_strip)
        
    cleaned = " ".join(lines).strip()
    return cleaned if cleaned else text


def generate_final_explanation(query: str, retrieved_wine: dict, history: List[Message] = []) -> str:
    """Uses RAG to generate the final sommelier explanation based on the actual retrieved wine."""
    if model is None or tokenizer is None:
        print("Warning: LLM not loaded. Using Mock Explanation.")
        return (f"Based on your request, I recommend the {retrieved_wine.get('title','wine')}. "
                f"This is a wonderful {retrieved_wine.get('variety','')} from {retrieved_wine.get('country','')}. "
                f"With its key characteristics: {str(retrieved_wine.get('description',''))[:180]}... "
                f"This bottle will be an excellent choice for you!")

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a Master Sommelier. Suggest the wine below to the customer based on their request. "
        f"Write in natural, warm, professional, and engaging English. "
        f"Briefly and persuasively explain why this wine is a match, and suggest a food pairing if appropriate.\n\n"
        f"Proposed Wine Details:\n"
        f"- Name: {retrieved_wine.get('title','')}\n"
        f"- Description: {str(retrieved_wine.get('description',''))[:300]}\n"
        f"<|eot_id|>"
    )

    for msg in history:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    # Pre-populate thought block to bypass LLM thinking time
    prompt += "<thought> I will write a warm recommendation for the selected wine. </thought>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=80, use_cache=True)
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return _clean_llm(response)

@app.post("/recommend")
def recommend_wine(request: QueryRequest):
    try:
        import string
        # Clean punctuation for intent matching
        clean_q = request.query.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        
        # 1. Intent Detection (Mock rule-based)
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "chào", "xin chào"]
        thanks = ["thanks", "thank you", "cảm ơn", "cam on"]
        
        # Is it a simple greeting?
        if any(g == clean_q or clean_q.startswith(g + " ") for g in greetings) and len(clean_q.split()) < 4:
            return {
                "type": "chat",
                "message": "Hello! I am your AI Sommelier. How can I help you today? Are you looking for a red, white, or perhaps a sparkling wine?"
            }
            
        # Is it a thank you?
        if any(t == clean_q or clean_q.startswith(t + " ") for t in thanks) and len(clean_q.split()) < 4:
            return {
                "type": "chat",
                "message": "You're very welcome! If you need any more recommendations or have questions about wine varieties, feel free to ask."
            }
            
        # If it's too short and not a greeting/thanks, ask for more details
        if len(clean_q.split()) < 3:
            return {
                "type": "chat",
                "message": "Please provide a few more details (e.g., wine type, region, budget, or food pairings) so I can find the perfect bottle for you."
            }

        # 2. Recommendation Phase
        # Phase 1: Generate ideal profile / ID
        print(f"Generating profile for query: {request.query}")
        ideal_profile_thought = generate_profile(request.query, request.history)
        
        # Phase 2: Hybrid Retrieval — ChromaDB (text) + GNN (structural)
        search_query = f"{request.query} {ideal_profile_thought}"
        retrieval_method = "chroma_only"   # default

        # ── 2a. ChromaDB text-similarity retrieval ────────────────────────
        print(f"Querying ChromaDB...")
        n_chroma_candidates = 10  # fetch more for fusion
        results = collection.query(
            query_texts=[search_query],
            n_results=n_chroma_candidates,
            include=["documents", "metadatas", "distances"],
        )
        
        if not results['documents'][0]:
            return {"error": "No matching wines found in inventory."}

        # Build candidate pool from ChromaDB results
        # ChromaDB distances are L2; convert to a similarity in [0, 1]
        chroma_candidates = {}  # key = title, value = dict
        chroma_dists = results.get("distances", [[]])[0]
        for rank_i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                chroma_dists if chroma_dists else [0.0] * len(results["documents"][0]),
            )
        ):
            # Normalise: score = 1 / (1 + dist) → higher is better
            chroma_score = 1.0 / (1.0 + float(dist))
            key = meta.get("title", f"chroma_{rank_i}")
            chroma_candidates[key] = {
                "title"       : meta.get("title", ""),
                "country"     : meta.get("country", ""),
                "variety"     : meta.get("variety", ""),
                "description" : doc,
                "chroma_score": chroma_score,
            }

        # Sort by chroma_score descending and pick the best
        ranked = sorted(
            chroma_candidates.values(),
            key=lambda c: c["chroma_score"],
            reverse=True,
        )
        best = ranked[0]

        retrieved_metadata = {
            "title"      : best["title"],
            "country"    : best["country"],
            "variety"    : best["variety"],
            "description": best["description"],
        }

        print(
            f"[RAG] Best match: {best['title']}  "
            f"(chroma={best['chroma_score']:.3f}, method={retrieval_method})"
        )
        
        # Phase 3: Generate Final Explanation
        print(f"Generating final explanation for {retrieved_metadata['title']}...")
        final_explanation = generate_final_explanation(request.query, retrieved_metadata, request.history)
        
        # Clean up any generated <thought> block from the final explanation shown to user
        if "</thought>" in final_explanation:
            final_explanation = final_explanation.split("</thought>")[-1].strip()
        
        # ── Phase 4: SHAP-based transparent scoring attribution ──────────
        xai_result = None
        if XAI_AVAILABLE and scoring_explainer is not None:
            try:
                t_xai = time.time()
                xai_result = scoring_explainer.explain(
                    request.query,
                    retrieved_metadata,
                    n_shap_samples=32,
                )
                print(f"SHAP explanation done in {xai_result['latency_ms']:.0f}ms")
            except Exception as e:
                print(f"[WARN] SHAP explanation failed: {e}")

        # ── Phase 5: Attention-based LLM explainability ─────────────────
        attention_result = None
        if ATTENTION_XAI_AVAILABLE and model is not None and tokenizer is not None:
            try:
                # Build the same prompt the LLM saw for the final explanation
                attn_prompt = (
                    f"Recommend wine: {request.query}. "
                    f"Wine: {retrieved_metadata.get('title', '')}. "
                    f"Variety: {retrieved_metadata.get('variety', '')}. "
                    f"Country: {retrieved_metadata.get('country', '')}."
                )
                attention_result = extract_attention_highlights(
                    model, tokenizer, attn_prompt, top_k=10,
                )
                print(f"Attention extraction done in {attention_result['latency_ms']:.0f}ms")
            except Exception as e:
                print(f"[WARN] Attention extraction failed: {e}")

        return {
            "type"                    : "recommendation",
            "message"                 : final_explanation,
            "retrieved_wine"          : retrieved_metadata,
            "generated_profile_thought": ideal_profile_thought,
            "xai_explanation"         : xai_result,
            "explanation_type"        : "transparent_scoring",
            "disclaimer"              : (
                "The SHAP explanation covers a transparent heuristic scoring "
                "function (4 hand-crafted features with fixed weights). "
                "It does NOT explain the LLM's internal reasoning or the "
                "vector-retrieval ranking. See 'attention_highlights' for "
                "token-level LLM focus."
            ),
            "attention_highlights"    : attention_result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Extra endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """System status — useful for demo and evaluation logging."""
    return {
        "status"         : "ok",
        "llm_loaded"     : model is not None,
        "chromadb_wines" : collection.count(),
        "xai_available"  : XAI_AVAILABLE and shap_background is not None,
        "shap_bg_shape"  : list(shap_background.shape) if shap_background is not None else None,
        "gnn_available"  : gnn_retriever is not None,
        "cv_available"   : CV_AVAILABLE and wine_scanner is not None,
        "ocr_ready"      : wine_scanner.ocr_available  if wine_scanner else False,
        "yolo_ready"     : wine_scanner.yolo_available if wine_scanner else False,
        "catalog_size"   : len(catalog_df_cache) if catalog_df_cache is not None else 0,
    }



# ══════════════════════════════════════════════════════════════════════════════
# CV ENDPOINTS — Wine Label Scanning
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/scan_label")
async def scan_label(file: UploadFile = File(...)):
    """
    Main CV endpoint for wine consultant.

    Workflow:
      1. Read uploaded image
      2. OCR → extract title, variety, country, year, price
      3. Fuzzy-search the catalog
         • FOUND  → return wine details + rich tasting notes + aroma profile
         • NOT FOUND → return extracted info + similar wines ranked by
                       country match → variety match → price proximity

    Request (multipart/form-data):
        file: image file (JPEG/PNG)

    Response:
        status        : "in_stock" | "not_in_stock" | "error"
        scan_info     : extracted label info
        wine          : matched wine dict (if in_stock)
        tasting_notes : sommelier paragraph (if in_stock)
        aroma_profile : aroma dict (if in_stock)
        food_pairings : list of foods (if in_stock)
        similar_wines : list of alternatives (if not_in_stock)
    """
    if not CV_AVAILABLE or wine_scanner is None:
        raise HTTPException(status_code=503, detail="CV module not available. Install: pip install easyocr Pillow")
    if catalog_df_cache is None:
        raise HTTPException(status_code=503, detail="Wine catalog not loaded. Check data/raw/winemag-data-130k-v2.csv")

    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted (JPEG, PNG, WEBP)")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        t_total = time.time()

        # ── Step 1: Scan the label ────────────────────────────────────────────
        scan_info = wine_scanner.scan(image_bytes)

        if "error" in scan_info:
            return {"status": "error", "detail": scan_info["error"]}
            
        # ── Step 1.5: LLM Extraction ─────────────────────────────────────────
        # Use LLM to re-parse raw OCR for maximum accuracy
        raw_ocr = scan_info.get("raw_ocr", "")
        if raw_ocr and model is not None and tokenizer is not None:
            print("[LLM] Enhancing OCR parse with Llama-3...")
            prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are an expert Wine Label Parser. Given raw OCR text from a wine bottle, extract the structured information. "
                "Return ONLY a pure JSON object (no markdown, no comments) with these keys: "
                '"title" (Wine name/brand), "variety" (Grape variety), "country" (Origin country), "year" (Vintage year, string), "price" (Price if present). '
                "If a field is missing, set it to null. Fix any obvious OCR typos in the variety.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"Raw OCR:\n{raw_ocr}\n<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n{"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                # Disable LoRA so it acts as standard LLaMA-3 for JSON extraction
                if hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        outputs = model.generate(
                            **inputs, max_new_tokens=150, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id
                        )
                else:
                    outputs = model.generate(
                        **inputs, max_new_tokens=150, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id
                    )
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            generated = "{" + generated.strip()
            
            try:
                import json
                clean_json = generated.replace('```json', '').replace('```', '').strip()
                idx = clean_json.rfind('}')
                if idx != -1:
                    clean_json = clean_json[:idx+1]
                llm_data = json.loads(clean_json)
                
                # Override the old rule-based parser results if LLM found something
                for k in ["title", "variety", "country", "year", "price"]:
                    val = llm_data.get(k)
                    if val and str(val).strip() and str(val).lower() != "null":
                        scan_info[k] = str(val).strip()
                        scan_info.setdefault("field_confidence", {})[k] = 0.95
                        
                print(f"[LLM] Parser Result: {llm_data}")
            except Exception as e:
                print(f"[LLM] Parser JSON Error: {e} -> Raw: {generated}")

        # Remove non-serialisable visual_embedding from response
        scan_info_out = {k: v for k, v in scan_info.items() if k != "visual_embedding"}

        # ── Step 2: Look up in catalog ────────────────────────────────────────
        lookup = find_in_catalog(scan_info, catalog_df_cache)

        if lookup["found"]:
            # ────────────────────────────────── FOUND: wine is in inventory ──
            wine = lookup["wine"]
            variety = wine.get("variety", scan_info.get("variety", ""))

            tasting_notes = build_tasting_notes(wine)
            aroma         = get_aroma_profile(variety)
            pairings      = get_food_pairings(variety)

            # Generate LLM explanation if model available
            if model is not None and tokenizer is not None:
                try:
                    llm_text = generate_final_explanation(
                        f"Tell me about {wine.get('title', 'this wine')}",
                        wine,
                    )
                    if llm_text:
                        tasting_notes = llm_text
                except Exception:
                    pass   # fallback to rule-based

            return {
                "status"       : "in_stock",
                "scan_info"    : scan_info_out,
                "match_type"   : lookup["match_type"],
                "match_score"  : lookup["similarity"],
                "wine"         : wine,
                "tasting_notes": tasting_notes,
                "aroma_profile": aroma,
                "food_pairings": pairings,
                "total_ms"     : round((time.time() - t_total) * 1000, 1),
            }

        else:
            # ─────────────────────────── NOT FOUND: find similar alternatives ─
            similar = find_similar_wines(scan_info, catalog_df_cache, n=5)

            # Enrich each similar wine with aroma summary
            for w in similar:
                w["tasting_summary"] = build_tasting_notes(w)

            return {
                "status"       : "not_in_stock",
                "scan_info"    : scan_info_out,
                "message"      : _build_not_found_message(scan_info),
                "similar_wines": similar,
                "total_ms"     : round((time.time() - t_total) * 1000, 1),
            }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze_label")
async def analyze_label_only(file: UploadFile = File(...)):
    """
    Lightweight endpoint: only run OCR + parse, do NOT query catalog.
    Useful for quick label reading without full recommendation.
    """
    if not CV_AVAILABLE or wine_scanner is None:
        raise HTTPException(status_code=503, detail="CV module not available.")

    image_bytes = await file.read()
    scan_info   = wine_scanner.scan(image_bytes)
    # Strip non-serialisable embedding
    return {k: v for k, v in scan_info.items() if k != "visual_embedding"}


def _build_not_found_message(scan_info: dict) -> str:
    """Build a human-friendly message for the not-in-stock case."""
    parts = ["This wine is not in our inventory."]
    v = scan_info.get("variety")
    c = scan_info.get("country")
    if v and c:
        parts.append(f"Based on the label, it appears to be a **{v}** from **{c}**.")
    elif v:
        parts.append(f"Based on the label, it appears to be a **{v}**.")
    elif c:
        parts.append(f"Based on the label, it appears to be from **{c}**.")
    parts.append("But we have similar bottles like this:")
    return " ".join(parts)


class ExplainRequest(BaseModel):
    query : str
    title : str
    price : Optional[float] = None
    variety: str = ""
    country: str = ""
    description: str = ""


@app.post("/explain")
def explain_wine(req: ExplainRequest):
    """
    Standalone heuristic feature attribution endpoint.
    Accepts a (query, wine) pair and returns feature-level attributions.
    Useful for the demo UI and evaluation logging.
    """
    if not XAI_AVAILABLE or scoring_explainer is None:
        raise HTTPException(status_code=503,
                            detail="XAI not available. Install shap and restart.")
    wine = {
        "title"      : req.title,
        "price"      : req.price,
        "variety"    : req.variety,
        "country"    : req.country,
        "description": req.description,
    }
    try:
        result = scoring_explainer.explain(req.query, wine, n_shap_samples=32)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
