import sys, os
_p = __import__('pathlib').Path(__file__).resolve()
sys.path.insert(0, str(_p.parents[1]))
sys.path.insert(0, str(_p.parent))
import config as cfg
from fastapi import FastAPI, HTTPException
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

# ── XAI module (Tuần 17) ─────────────────────────────────────────────────────
try:
    from xai_shap import (
        explain_recommendation,
        build_background,
        extract_features,
        FEATURE_NAMES,
    )
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("[WARN] xai_shap not found — XAI disabled. Run: pip install shap")

app = FastAPI(title="Wine Recommendation API")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=str(cfg.CHROMA_DB))
collection = chroma_client.get_or_create_collection(name="wine_inventory")

# Global Variables
tokenizer        = None
model            = None
shap_background  = None   # np.ndarray (n_bg, 5) for SHAP KernelExplainer
catalog_df_cache = None   # cached DataFrame for SHAP background builds

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
    global tokenizer, model, shap_background, catalog_df_cache

    # ── 1. Populate Vector DB ─────────────────────────────────────────────────
    print("Initializing Vector DB Inventory...")
    try:
        if collection.count() > 0:
            print(f"Vector DB already has {collection.count()} wines. Skipping ingestion.")
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, str(cfg.WINE_CSV))
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=["country","variety","description","title","price"]).head(5000)
            docs, metadatas, ids = [], [], []
            for idx, row in df.iterrows():
                docs.append(row["description"])
                metadatas.append({
                    "title"  : row["title"],
                    "country": row["country"],
                    "variety": row["variety"],
                    "price"  : float(row["price"]),
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

    # ── 3. Build SHAP background (Tuần 17 — XAI) ─────────────────────────────
    if XAI_AVAILABLE:
        try:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    str(cfg.WINE_CSV))
            catalog_df_cache = pd.read_csv(csv_path).dropna(
                subset=["country","variety","description","price"]
            )
            shap_background = build_background(catalog_df_cache, n_samples=100)
            print(f"SHAP background built: {shap_background.shape}")
        except Exception as e:
            print(f"[WARN] SHAP background build failed: {e}")
    else:
        print("[INFO] XAI disabled (xai_shap module not available).")

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

    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a Master Sommelier. Analyze the user's request and determine the ideal structural profile of the wine. Then, output the Semantic ID of the perfect match, followed by a persuasive explanation.<|eot_id|>"
    
    for msg in history:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>"
        
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n<thought>"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, stop_strings=["</thought>"], tokenizer=tokenizer)
        
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return "<thought>" + generated_text

def generate_final_explanation(query: str, retrieved_wine: dict, history: List[Message] = []) -> str:
    """Uses RAG to generate the final sommelier explanation based on the actual retrieved wine."""
    if model is None or tokenizer is None:
        print("Warning: LLM not loaded. Using Mock Explanation.")
        return f"Based on your request for '{query}', I highly recommend the {retrieved_wine['title']}. The combination of {retrieved_wine['variety']} from {retrieved_wine['country']} offers notes of: {retrieved_wine['description']}. This is a perfect match for what you are looking for!"

    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a Master Sommelier. Recommend the following wine based on the user's query. Explain the pairing and flavor profile persuasively.\n\nAvailable Wine: {retrieved_wine['title']}\nPrice: ${retrieved_wine['price']}\nNotes: {retrieved_wine['description']}\n<|eot_id|>"
    
    for msg in history:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>"
        
    prompt += f"<|start_header_id|>user<|end_header_id|>\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response

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
                "message": "Hello! I am your AI Sommelier. What kind of wine are you looking for today? Do you prefer red, white, or perhaps a sweet dessert wine?"
            }
            
        # Is it a thank you?
        if any(t == clean_q or clean_q.startswith(t + " ") for t in thanks) and len(clean_q.split()) < 4:
            return {
                "type": "chat",
                "message": "You're very welcome! Let me know if you need another recommendation or have any questions about wine."
            }
            
        # If it's too short and not a greeting/thanks, ask for more details
        if len(clean_q.split()) < 3:
            return {
                "type": "chat",
                "message": "Could you provide a bit more detail? For example, are you looking for a red or white wine, and what is your price range?"
            }

        # 2. Recommendation Phase
        # Phase 1: Generate ideal profile / ID
        print(f"Generating profile for query: {request.query}")
        ideal_profile_thought = generate_profile(request.query, request.history)
        
        # Phase 2: RAG / Vector Search based on the query and thought
        search_query = f"{request.query} {ideal_profile_thought}"
        print(f"Querying ChromaDB...")
        results = collection.query(
            query_texts=[search_query],
            n_results=1
        )
        
        if not results['documents'][0]:
            return {"error": "No matching wines found in inventory."}
            
        retrieved_metadata = results['metadatas'][0][0]
        retrieved_metadata['description'] = results['documents'][0][0]
        
        # Phase 3: Generate Final Explanation
        print(f"Generating final explanation for {retrieved_metadata['title']}...")
        final_explanation = generate_final_explanation(request.query, retrieved_metadata, request.history)
        
        # Clean up any generated <thought> block from the final explanation shown to user
        if "</thought>" in final_explanation:
            final_explanation = final_explanation.split("</thought>")[-1].strip()
        
        # ── Phase 4: SHAP-based heuristic feature attribution ────────────
        xai_result = None
        if XAI_AVAILABLE and shap_background is not None:
            try:
                t_xai = time.time()
                xai_result = explain_recommendation(
                    request.query,
                    retrieved_metadata,
                    shap_background,
                    n_shap_samples=64,
                )
                print(f"SHAP explanation done in {xai_result['latency_ms']:.0f}ms")
            except Exception as e:
                print(f"[WARN] SHAP explanation failed: {e}")

        return {
            "type"                    : "recommendation",
            "message"                 : final_explanation,
            "retrieved_wine"          : retrieved_metadata,
            "generated_profile_thought": ideal_profile_thought,
            "xai_explanation"         : xai_result,   # Heuristic SHAP feature attribution
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
    }


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
    if not XAI_AVAILABLE or shap_background is None:
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
        result = explain_recommendation(req.query, wine, shap_background,
                                        n_shap_samples=64)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
