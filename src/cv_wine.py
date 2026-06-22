"""
cv_wine.py — Wine Label Scanner (Computer Vision Module)
=========================================================
Pipeline 2 tầng:
  Stage 1 (YOLO)   → Phát hiện & crop vùng nhãn trong ảnh chai nguyên
  Stage 2 (EasyOCR)→ Đọc text từ ảnh đã crop + enhance
  Stage 3 (Parser) → Extract: title, variety, country, year, price
  Stage 4 (CLIP)   → Visual embedding (optional, fallback)

Khi không có YOLO (không cài / không detect được label)
  → tự động fallback sang full-image OCR.

Usage:
    scanner = WineLabelScanner()
    scanner.load()                    # khởi động tất cả models
    result  = scanner.scan(bytes)     # {title, variety, country, year, price, …}
"""

import re
import sys
import time
import warnings
import io
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))

import numpy as np
import pandas as pd

# ─── Wine Knowledge Base ──────────────────────────────────────────────────────

WINE_VARIETIES = [
    # Reds
    "Cabernet Sauvignon", "Merlot", "Pinot Noir", "Syrah", "Shiraz",
    "Malbec", "Zinfandel", "Sangiovese", "Tempranillo", "Grenache",
    "Nebbiolo", "Barbera", "Montepulciano", "Carmenere", "Petite Sirah",
    "Mourvèdre", "Mourvedre", "Touriga Nacional", "Tannat", "Primitivo",
    "Red Blend", "Meritage", "GSM", "Cabernet Franc", "Petit Verdot",
    "Garnacha", "Cinsault", "Nero d'Avola", "Aglianico",
    # Whites
    "Chardonnay", "Sauvignon Blanc", "Riesling", "Pinot Grigio",
    "Pinot Gris", "Gewurztraminer", "Viognier", "Chenin Blanc",
    "Albariño", "Albarino", "Gruner Veltliner", "Torrontés",
    "Vermentino", "Fiano", "Greco", "Falanghina", "Trebbiano",
    "White Blend", "Muscat", "Moscato",
    # Sparkling / Dessert / Fortified
    "Champagne", "Prosecco", "Cava", "Crémant", "Sekt",
    "Port", "Sherry", "Madeira", "Marsala", "Vin Doux",
    # Rosé
    "Rosé", "Rose", "Blush",
    # Regional/Blend names often on labels
    "Bordeaux", "Burgundy", "Chianti", "Barolo", "Barbaresco",
    "Rioja", "Ribera del Duero", "Amarone", "Brunello",
    "Valpolicella", "Soave", "Greco di Tufo",
]

# Sort by length descending so longer names match first
WINE_VARIETIES_SORTED = sorted(WINE_VARIETIES, key=len, reverse=True)

COUNTRY_KEYWORDS: dict[str, list[str]] = {
    "France": [
        "france", "french", "bordeaux", "burgundy", "bourgogne",
        "champagne", "alsace", "rhône", "rhone", "loire",
        "languedoc", "provence", "roussillon",
    ],
    "Italy": [
        "italy", "italian", "tuscany", "toscana", "piemonte",
        "veneto", "sicilia", "sicily", "barolo", "chianti",
        "amarone", "brunello", "friuli", "lombardia",
    ],
    "Spain": [
        "spain", "spanish", "españa", "rioja", "ribera",
        "penedès", "priorat", "cava", "galicia", "rueda",
    ],
    "US": [
        "california", "napa", "sonoma", "oregon", "washington",
        "united states", "american", "usa", "paso robles",
        "monterey", "columbia valley",
    ],
    "Argentina": [
        "argentina", "mendoza", "malbec", "argentine", "patagonia",
        "salta", "san juan",
    ],
    "Australia": [
        "australia", "barossa", "australian", "hunter valley",
        "mclaren vale", "yarra valley", "margaret river", "clare valley",
    ],
    "Chile": [
        "chile", "chilean", "maipo", "casablanca", "colchagua",
        "aconcagua", "limari",
    ],
    "Germany": [
        "germany", "german", "mosel", "rheingau", "pfalz",
        "rheinhessen", "nahe", "ahr", "deutsch",
    ],
    "Portugal": [
        "portugal", "portuguese", "douro", "alentejo", "vinho verde",
        "dao", "bairrada", "porto",
    ],
    "New Zealand": [
        "new zealand", "marlborough", "hawke", "central otago",
        "gisborne", "waipara",
    ],
    "South Africa": [
        "south africa", "stellenbosch", "cape", "franschhoek",
        "paarl", "swartland",
    ],
    "Austria": ["austria", "austrian", "grüner", "gruner", "wachau", "burgenland"],
}

FOOD_PAIRINGS: dict[str, list[str]] = {
    "Cabernet Sauvignon": ["steak", "lamb", "beef", "hard cheese", "dark chocolate"],
    "Merlot": ["pasta", "pizza", "chicken", "grilled vegetables", "soft cheese"],
    "Pinot Noir": ["salmon", "duck", "mushroom", "tuna", "charcuterie"],
    "Chardonnay": ["seafood", "lobster", "chicken", "cream pasta", "brie"],
    "Sauvignon Blanc": ["salad", "oyster", "goat cheese", "sushi", "light fish"],
    "Malbec": ["steak", "bbq", "beef", "burger", "hard cheese"],
    "Riesling": ["spicy food", "thai", "asian", "pork", "sushi"],
    "Syrah": ["lamb", "bbq", "game", "stew", "aged cheese"],
    "Pinot Grigio": ["light fish", "seafood", "salad", "antipasti", "prosciutto"],
    "Tempranillo": ["tapas", "lamb", "roast pork", "manchego", "paella"],
    "Sangiovese": ["pizza", "pasta", "tomato sauce", "antipasto", "prosciutto"],
    "Champagne": ["oysters", "caviar", "sushi", "light appetizers", "fried food"],
    "Moscato": ["dessert", "fruit tart", "cheese", "light pastry"],
    "Port": ["blue cheese", "chocolate", "dessert", "nuts", "stilton"],
}

AROMA_PROFILES: dict[str, dict] = {
    "Cabernet Sauvignon": {
        "primary": ["blackcurrant", "blackberry", "plum", "cedar"],
        "secondary": ["tobacco", "leather", "graphite", "vanilla"],
        "tertiary": ["cigar box", "truffle", "dried fruit"],
        "palate": "Full-bodied with firm tannins, long finish",
        "color": "Deep ruby to garnet",
    },
    "Merlot": {
        "primary": ["plum", "cherry", "blueberry", "chocolate"],
        "secondary": ["bay leaf", "vanilla", "mocha", "coffee"],
        "tertiary": ["dried fruit", "leather", "earth"],
        "palate": "Medium-full body, soft tannins, velvety texture",
        "color": "Medium to deep ruby",
    },
    "Pinot Noir": {
        "primary": ["cherry", "raspberry", "strawberry", "rose petal"],
        "secondary": ["mushroom", "forest floor", "vanilla", "spice"],
        "tertiary": ["truffle", "leather", "game", "balsamic"],
        "palate": "Light to medium body, silky tannins, elegant finish",
        "color": "Translucent ruby to garnet",
    },
    "Chardonnay": {
        "primary": ["green apple", "lemon", "pear", "pineapple"],
        "secondary": ["butter", "cream", "vanilla", "toast"],
        "tertiary": ["honey", "nuts", "brioche"],
        "palate": "Medium to full body, refreshing acidity",
        "color": "Pale to golden yellow",
    },
    "Sauvignon Blanc": {
        "primary": ["grapefruit", "lime", "passionfruit", "green bell pepper"],
        "secondary": ["grass", "herbs", "gooseberry", "elderflower"],
        "tertiary": ["smoke", "flint", "mineral"],
        "palate": "Light to medium body, high acidity, crisp finish",
        "color": "Pale to medium yellow-green",
    },
    "Malbec": {
        "primary": ["blackberry", "plum", "black cherry", "violet"],
        "secondary": ["leather", "tobacco", "dark chocolate", "espresso"],
        "tertiary": ["dried fruit", "earth", "smoke"],
        "palate": "Full-bodied, rich, velvety with dark fruit finish",
        "color": "Deep purple to ruby",
    },
    "Riesling": {
        "primary": ["peach", "apricot", "lime", "green apple"],
        "secondary": ["honey", "beeswax", "floral", "white peach"],
        "tertiary": ["petrol", "slate", "mineral"],
        "palate": "Light body, vibrant acidity, lingering minerality",
        "color": "Pale to medium gold",
    },
    "Syrah": {
        "primary": ["blackberry", "black pepper", "olive", "blueberry"],
        "secondary": ["smoked meat", "leather", "dark chocolate", "tobacco"],
        "tertiary": ["coffee", "tar", "game"],
        "palate": "Full-bodied, robust tannins, savory and spicy finish",
        "color": "Deep, inky purple",
    },
    "Tempranillo": {
        "primary": ["cherry", "plum", "strawberry", "tomato"],
        "secondary": ["leather", "tobacco", "vanilla", "coconut"],
        "tertiary": ["cedar", "earth", "dried herbs"],
        "palate": "Medium to full body, moderate tannins, earthy finish",
        "color": "Medium to deep ruby",
    },
    "Sangiovese": {
        "primary": ["cherry", "red plum", "dried herb", "tomato"],
        "secondary": ["leather", "tobacco", "spice", "balsamic"],
        "tertiary": ["earth", "dried fruit", "iron"],
        "palate": "Medium body, firm acidity and tannins, savory finish",
        "color": "Medium garnet to ruby",
    },
}

# Regex patterns — multi-currency price, year, critic score
_PRICE_RE  = re.compile(
    r"(?:"
    r"(?:\$|USD|usd)\s*([\d,]+(?:\.\d{1,2})?)"
    r"|([\d,]+(?:\.\d{1,2})?)\s*(?:USD|usd)"
    r"|(?:€|EUR|eur)\s*([\d,]+(?:[.,]\d{1,2})?)"
    r"|([\d,]+(?:[.,]\d{1,2})?)\s*(?:€|EUR|eur)"
    r"|(?:£|GBP|gbp)\s*([\d,]+(?:\.\d{1,2})?)"
    r")"
)
_YEAR_RE   = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
_SCORE_RE  = re.compile(r"\b(\d{2})\s*(?:pts?|points?|pnts?|/100)\b", re.I)
_VOLUME_RE = re.compile(r"\b(\d{2,3})\s*(?:cl|ml|CL|ML)\b")  # ignore volume as price
_ALCOHOL_RE= re.compile(r"\b(\d{1,2}(?:\.\d)?%)\b")           # ignore alcohol %

# Noise words to skip in title extraction
_TITLE_SKIP = {
    "estate","bottled","reserve","selected","selection","winery","chateau",
    "domaine","bodega","cantina","fattoria","quinta","mas","clos","cave",
    "docg","doc","aoc","ava","igt","vdp","qba","qmp","dop","igp",
    "red","white","rose","rose","wine","vintage","bottling","vieilles",
    "vines","vignes","grand","cru","premier","superieur","superiore",
}


def check_cuda_working():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        x = torch.zeros(1, device="cuda")
        y = x + 1
        _ = y.cpu()
        return True
    except Exception as e:
        print(f"[CV CUDA Check] CUDA available but failed kernel execution: {e}. Falling back to CPU.")
        return False

# ─── Main Scanner Class ───────────────────────────────────────────────────────

class WineLabelScanner:
    """
    Wine label scanner v2 — YOLO + EasyOCR + CLIP pipeline.

    Stage 1: YOLOv8n phát hiện vùng nhãn / chai trong ảnh đầu vào.
    Stage 2: EasyOCR đọc text từ ảnh đã crop + được enhance.
    Stage 3: Regex/NLP parser → structured wine info.
    Stage 4: CLIP embedding (optional, lazy-load).

    Nếu YOLO không được cài hoặc không detect được gì → full-image OCR.
    """

    # COCO class IDs mà YOLOv8 pretrained có thể detect được,
    # dùng làm proxy cho chai rượu:
    # 39=bottle, 75=vase, 73=book (label-like shape), 76=scissors
    _YOLO_LABEL_CLASSES = {39, 75, 73}    # bottle, vase, book
    # Nếu không được class nào trong số trên, lấy box có diện tích lớn nhất
    _YOLO_MIN_CONF = 0.25
    _YOLO_MIN_AREA_RATIO = 0.04   # bỏ qua box nhỏ hơn 4% ảnh

    def __init__(self):
        self._ocr_reader     = None
        self._clip_model     = None
        self._clip_processor = None
        self._yolo_model     = None
        self.vlm = None
        self.ocr_available   = False
        self.clip_available  = False
        self.yolo_available  = False

    # ── Model Loading ─────────────────────────────────────────────────

    def load(self, load_clip: bool = False):
        """Pre-load OCR + YOLO (and optionally CLIP). Call once at startup."""
        self._init_ocr()
        self._init_yolo()
        if load_clip:
            self._init_clip()

    def _init_ocr(self) -> bool:
        try:
            from src.vlm_parser import florence_parser
            self.vlm = florence_parser
            # Attempt to preload
            self.vlm.load()
            self.ocr_available = True
            print("[CV] Florence-2 VLM ready.")
            return True
        except Exception as exc:
            print(f"[CV] Failed to load VLM OCR: {exc}")
            self.ocr_available = False

        if self._ocr_reader is not None:
            return True
        try:
            import easyocr
            print("[CV] Loading EasyOCR (English)…")
            t0 = time.time()
            self._ocr_reader = easyocr.Reader(["en"], verbose=False)
            print(f"[CV] EasyOCR ready in {time.time()-t0:.1f}s")
            self.ocr_available = True
            return True
        except ImportError:
            print("[CV] EasyOCR not installed. `pip install easyocr`")
            return False
        except Exception as exc:
            print(f"[CV] EasyOCR failed to load: {exc}")
            return False

    def _init_yolo(self) -> bool:
        """Lazy-load YOLO model. Prefers fine-tuned yolov8_wine.pt over yolov8n.pt."""
        if self._yolo_model is not None:
            return True
        try:
            from ultralytics import YOLO
            import os
            t0 = time.time()
            
            # Check for fine-tuned weights in models/
            ft_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo11_wine.pt")
            if os.path.exists(ft_path):
                print(f"[CV] Loading fine-tuned YOLO model: {ft_path}…")
                self._yolo_model = YOLO(ft_path)
                self.is_fine_tuned = True
            else:
                # Fallback to YOLOv8 fine-tuned if available
                v8_ft_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolov8_wine.pt")
                if os.path.exists(v8_ft_path):
                    print(f"[CV] Loading fine-tuned YOLOv8 model: {v8_ft_path}…")
                    self._yolo_model = YOLO(v8_ft_path)
                    self.is_fine_tuned = True
                else:
                    print("[CV] Loading base YOLO11n for label detection…")
                    self._yolo_model = YOLO("yolo11n.pt")
                    self.is_fine_tuned = False
                
            self._yolo_model.overrides["verbose"] = False
            print(f"[CV] YOLO model ready in {time.time()-t0:.1f}s | Fine-tuned: {getattr(self, 'is_fine_tuned', False)}")
            self.yolo_available = True
            return True
        except ImportError:
            print("[CV] ultralytics not installed — YOLO disabled. `pip install ultralytics`")
            return False
        except Exception as exc:
            print(f"[CV] YOLO load failed: {exc}")
            return False

    def _init_clip(self) -> bool:
        if self._clip_model is not None:
            return True
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            device = "cuda" if check_cuda_working() else "cpu"
            print(f"[CV] Loading CLIP in FP16 on {device}…")
            t0 = time.time()
            try:
                self._clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
                if device == "cuda":
                    self._clip_model = CLIPModel.from_pretrained(
                        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", 
                        torch_dtype=torch.float16
                    ).eval().to(device)
                else:
                    self._clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").eval()
            except Exception as e:
                print(f"[CV] laion CLIP load failed: {e}. Falling back to openai CLIP...")
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                if device == "cuda":
                    self._clip_model = CLIPModel.from_pretrained(
                        "openai/clip-vit-large-patch14", 
                        torch_dtype=torch.float16
                    ).eval().to(device)
                else:
                    self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
            print(f"[CV] CLIP ready in {time.time()-t0:.1f}s")
            self.clip_available = True
            return True
        except Exception as exc:
            print(f"[CV] CLIP failed to load: {exc}")
            return False

    def load_catalog_embeddings(self, catalog_df):
        """
        Offline Stage: Precompute or load the product catalog image/text embeddings
        into a static 2D tensor matrix stored on GPU memory.
        """
        import torch
        import os
        device = "cuda" if check_cuda_working() else "cpu"
        self._init_clip()
        
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "catalog_clip_embeddings.pt")
        
        if os.path.exists(cache_path):
            print(f"[CV] Loading pre-computed catalog embeddings from {cache_path}...")
            try:
                self.catalog_embeds = torch.load(cache_path).to(device)
                if device == "cuda":
                    self.catalog_embeds = self.catalog_embeds.to(torch.float16)
                print(f"[CV] Catalog embeddings loaded. Shape: {self.catalog_embeds.shape}")
                return
            except Exception as e:
                print(f"[CV] Failed to load catalog embeddings cache: {e}. Recomputing...")

        # Recompute on the fly
        print("[CV] Pre-computing catalog embeddings (Offline Stage simulation)...")
        if self._clip_model is None:
            print("[CV] CLIP not loaded. Cannot precompute embeddings.")
            return

        texts = []
        for idx, row in catalog_df.iterrows():
            desc = f"A wine label showing {row.get('title', row.get('name', ''))}. It is a {row.get('variety', row.get('type', ''))} from {row.get('country', row.get('brand', ''))}."
            texts.append(desc[:120])

        embeds = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_emb = self._clip_embed_texts(batch)
            if batch_emb is not None:
                embeds.append(torch.tensor(batch_emb))
            else:
                embeds.append(torch.zeros((len(batch), 512)))
        
        self.catalog_embeds = torch.cat(embeds, dim=0).to(device)
        if device == "cuda":
            self.catalog_embeds = self.catalog_embeds.to(torch.float16)
            
        try:
            torch.save(self.catalog_embeds.cpu(), cache_path)
            print(f"[CV] Saved pre-computed catalog embeddings to {cache_path}")
            self.catalog_embeds = self.catalog_embeds.to(device)
        except Exception as e:
            print(f"[WARN] Failed to cache catalog embeddings: {e}")

        print(f"[CV] Catalog embeddings matrix ready on {device}. Shape: {self.catalog_embeds.shape}")

    # ── Public API ──────────────────────────────────────────────────

    def scan(self, image_bytes: bytes) -> dict:
        """
        Main entry point — 5-stage CV pipeline.

        Stage 1: YOLO — detect & crop label region
        Stage 1.5: Cylindrical Unwarping — correct curvature distortion
        Stage 2: Enhance — sharpen, contrast, upscale
        Stage 3: Multi-pass EasyOCR/Florence-2 — enhanced + grayscale + rotations
        Stage 4: Parser — variety(fuzzy), country, year, price(multi-currency)
        Stage 4.5: VLM Captioner — extract label visual style (graphics on label)
        Stage 5: CLIP — Visual embedding (optional, fallback)

        Returns dict with keys:
            raw_ocr, title, variety, country, year, price, score_points,
            wine_color, label_style, confidence, field_confidence, yolo_used, crop_box,
            annotated_image_b64, all_ocr_lines, latency_ms
        """
        t0 = time.time()
        from PIL import Image

        # Load image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            return {"error": f"Cannot open image: {exc}", "confidence": 0.0}

        orig_img = pil_img.copy()

        # ── Stage 1: YOLO detection ───────────────────────────────────────
        if not self.yolo_available:
            self._init_yolo()

        all_boxes, crop_img, crop_box, yolo_used = self._detect_all_regions(pil_img)

        # ── Stage 1.5: Cylindrical Unwarping Preprocessing ─────────────────
        try:
            print("[CV] Applying cylindrical unwarping to correct bottle curvature...")
            crop_img = self.preprocess_cylindrical_image(crop_img, radius_ratio=0.85)
        except Exception as e:
            print(f"[CV] Cylindrical unwarping failed (fallback to original crop): {e}")

        # ── Stage 2: Enhance ─────────────────────────────────────────────
        enhanced_img, scale1 = self._enhance_for_ocr(crop_img)

        # ── Stage 3: Multi-pass OCR ───────────────────────────────────────
        if not self.ocr_available:
            self._init_ocr()

        offset1 = (crop_box[0], crop_box[1]) if crop_box else (0, 0)
        raw_ocr, all_ocr_lines = self._multipass_ocr(enhanced_img, crop_img, scale1, offset1)

        # If YOLO crop gave poor result, try full image
        if yolo_used and len(raw_ocr.strip()) < 10:
            full_enhanced, scale2 = self._enhance_for_ocr(pil_img)
            # Also apply cylindrical unwarp on full image if needed, or fallback
            try:
                unwarped_full = self.preprocess_cylindrical_image(pil_img, radius_ratio=0.85)
                full_enhanced, scale2 = self._enhance_for_ocr(unwarped_full)
            except:
                pass
            raw_full, lines_full = self._multipass_ocr(full_enhanced, pil_img, scale2, (0,0))
            if len(raw_full) > len(raw_ocr):
                raw_ocr      = raw_full
                all_ocr_lines= lines_full
                yolo_used    = False
                crop_box     = None
                all_boxes    = []
                print("[CV] YOLO crop gave poor OCR — fell back to full unwarped image")

        # ── Stage 4: Parse ────────────────────────────────────────────────
        parsed = self._parse(raw_ocr)

        # ── Hierarchical Regional OCR Override ────────────────────────────
        if getattr(self, "is_fine_tuned", False) and yolo_used:
            print("[CV] Running Hierarchical Wine Label Parsing...")
            regional_parsed = {}
            for box in all_boxes:
                cls_name = box["cls_name"]
                if cls_name in ["brand", "vintage", "variety", "region"]:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    # Add margin/padding to the crop region
                    px = max(5, int((x2 - x1) * 0.05))
                    py = max(5, int((y2 - y1) * 0.05))
                    cx1 = max(0, x1 - px)
                    cy1 = max(0, y1 - py)
                    cx2 = min(pil_img.width, x2 + px)
                    cy2 = min(pil_img.height, y2 + py)
                    
                    crop_region = pil_img.crop((cx1, cy1, cx2, cy2))
                    
                    # Unwarp the region too
                    try:
                        crop_region = self.preprocess_cylindrical_image(crop_region, radius_ratio=0.85)
                    except:
                        pass
                    
                    if self.vlm is not None:
                        ocr_txt = self.vlm.ocr(crop_region)
                    elif self._ocr_reader is not None:
                        arr = np.array(crop_region)
                        res = self._ocr_reader.readtext(arr)
                        ocr_txt = " ".join([r[1] for r in res])
                    else:
                        ocr_txt = ""
                        
                    txt = ocr_txt.strip()
                    if txt:
                        if cls_name == "brand":
                            regional_parsed["title"] = txt
                        elif cls_name == "vintage":
                            regional_parsed["year"] = txt
                        elif cls_name == "variety":
                            regional_parsed["variety"] = txt
                        elif cls_name == "region":
                            regional_parsed["country"] = txt

            if regional_parsed:
                for field in ["title", "variety", "country", "year"]:
                    if regional_parsed.get(field):
                        parsed[field] = regional_parsed[field]
                print(f"[CV] Hierarchical regional override: {regional_parsed}")

        # ── Stage 4.5: VLM Captioner (Label Style) ──────────────────────────
        label_style = "N/A"
        if self.ocr_available and self.vlm is not None:
            try:
                print("[CV] Extracting label graphics and visual style...")
                label_style = self.vlm.caption(crop_img)
                print(f"[CV] Label style result: {label_style}")
            except Exception as e:
                print(f"[CV] Captioning task failed: {e}")

        wine_color = self._detect_color(crop_img)

        # ── Annotate image with YOLO boxes and OCR boxes ─────────────────
        annotated_b64 = self._annotate_image(orig_img, all_boxes, crop_box, all_ocr_lines)

        # Optional CLIP
        visual_emb = None
        if self.clip_available:
            visual_emb = self._clip_embed(orig_img)

        return {
            "raw_ocr"          : raw_ocr,
            "all_ocr_lines"    : all_ocr_lines,      # list of {text, conf}
            "title"            : parsed.get("title"),
            "variety"          : parsed.get("variety"),
            "country"          : parsed.get("country"),
            "year"             : parsed.get("year"),
            "price"            : parsed.get("price"),
            "score_points"     : parsed.get("score"),
            "wine_color"       : wine_color,
            "label_style"      : label_style,        # Visual style caption
            "confidence"       : round(parsed.get("confidence", 0.0), 2),
            "field_confidence" : parsed.get("field_confidence", {}),
            "yolo_used"        : yolo_used,
            "crop_box"         : crop_box,
            "all_boxes"        : all_boxes,
            "annotated_image_b64": annotated_b64,
            "visual_embedding" : visual_emb.tolist() if visual_emb is not None else None,
            "latency_ms"       : round((time.time() - t0) * 1000, 1),
        }

    # ── Stage 1: YOLO Detection ──────────────────────────────────────────

    def _detect_all_regions(self, pil_img):
        """
        Detect ALL YOLO boxes on image. Returns:
            (all_boxes_list, cropped_PIL, best_crop_box|None, yolo_used_bool)
        all_boxes = [{x1,y1,x2,y2,conf,cls,cls_name}, ...]
        """
        if not self.yolo_available or self._yolo_model is None:
            return [], pil_img, None, False

        # Keep for backward compatibility
        return self._detect_label_region_v2(pil_img)

    def _detect_label_region(self, pil_img):
        """Legacy wrapper — returns (crop, box, used)."""
        _, crop, box, used = self._detect_all_regions(pil_img)
        return crop, box, used

    def _detect_label_region_v2(self, pil_img):
        """
        Full detection: returns all boxes + best crop.
        """
        if not self.yolo_available or self._yolo_model is None:
            return [], pil_img, None, False

        try:
            W, H = pil_img.size
            min_area = W * H * self._YOLO_MIN_AREA_RATIO
            names   = self._yolo_model.names   # {id: class_name}

            results = self._yolo_model.predict(
                source=np.array(pil_img),
                conf=self._YOLO_MIN_CONF,
                verbose=False,
                device="cpu",
            )
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return [], pil_img, None, False

            bdata = boxes.data.cpu().numpy()   # (N, 6): x1,y1,x2,y2,conf,cls

            # Build structured all_boxes list
            all_boxes = []
            for row in bdata:
                cls_id = int(row[5])
                all_boxes.append({
                    "x1": int(row[0]), "y1": int(row[1]),
                    "x2": int(row[2]), "y2": int(row[3]),
                    "conf": round(float(row[4]), 3),
                    "cls" : cls_id,
                    "cls_name": names.get(cls_id, str(cls_id)),
                })

            # Select best crop box
            if getattr(self, "is_fine_tuned", False):
                # The label crop box is the minimum bounding box containing all detected regions
                x1 = int(np.min(bdata[:, 0]))
                y1 = int(np.min(bdata[:, 1]))
                x2 = int(np.max(bdata[:, 2]))
                y2 = int(np.max(bdata[:, 3]))
                best = [x1, y1, x2, y2, np.max(bdata[:, 4]), 0]
            else:
                preferred = bdata[np.isin(bdata[:,5].astype(int), list(self._YOLO_LABEL_CLASSES))]
                pool      = preferred if len(preferred) > 0 else bdata

                areas = (pool[:,2]-pool[:,0]) * (pool[:,3]-pool[:,1])
                valid = areas >= min_area
                pool  = pool[valid]
                if len(pool) == 0:
                    if len(bdata) > 0:
                        all_a = (bdata[:,2]-bdata[:,0])*(bdata[:,3]-bdata[:,1])
                        pool  = bdata[np.argmax(all_a):np.argmax(all_a)+1]
                    else:
                        return all_boxes, pil_img, None, False

                best = pool[np.argmax(pool[:,4])]
                x1, y1, x2, y2 = map(int, best[:4])

            pad_x = max(8, int((x2-x1)*0.05))
            pad_y = max(8, int((y2-y1)*0.05))
            x1 = max(0, x1-pad_x);  y1 = max(0, y1-pad_y)
            x2 = min(W, x2+pad_x);  y2 = min(H, y2+pad_y)

            crop = pil_img.crop((x1, y1, x2, y2))
            print(f"[CV] YOLO best crop: ({x1},{y1})->({x2},{y2})  "
                  f"conf={best[4]:.2f}  class={int(best[5])}  "
                  f"total_boxes={len(all_boxes)}")
            return all_boxes, crop, (x1, y1, x2, y2), True

        except Exception as exc:
            print(f"[CV] YOLO detection failed: {exc} -- using full image")
            return [], pil_img, None, False

    # ── Stage 2: Image Enhancement ─────────────────────────────────────

    def _enhance_for_ocr(self, pil_img):
        """
        Sharpen and improve contrast of the label image before OCR.
        Returns (enhanced_PIL_Image, scale_factor).
        """
        try:
            from PIL import Image as PILImage, ImageEnhance, ImageFilter
            scale = 1.0
            W, H = pil_img.size
            if min(W, H) < 200:
                scale = 200 / min(W, H)
                pil_img = pil_img.resize(
                    (int(W * scale), int(H * scale)),
                    resample=PILImage.LANCZOS if hasattr(PILImage, 'LANCZOS') else PILImage.BICUBIC,
                )
            img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img = img.filter(ImageFilter.MedianFilter(size=1))
            return img, scale
        except Exception:
            return pil_img, 1.0


    # ── Stage 3: EasyOCR ─────────────────────────────────────────────

    def _run_ocr_detail(self, pil_img, conf_thresh=0.2, scale=1.0, offset=(0,0)) -> tuple:
        """Run Florence VLM OCR, return (raw_text_str, ocr_lines_list)."""
        try:
            if not getattr(self, 'vlm', None):
                self._init_ocr()
                
            raw_lines = self.vlm.ocr_with_regions(pil_img)
            
            lines, texts = [], []
            for ln in raw_lines:
                texts.append(ln["text"])
                orig_bbox = [
                    [int((pt[0] / scale) + offset[0]), int((pt[1] / scale) + offset[1])]
                    for pt in ln["bbox"]
                ]
                lines.append({
                    "text": ln["text"],
                    "conf": ln["conf"],
                    "bbox": orig_bbox
                })
            return " | ".join(texts), lines
        except Exception as exc:
            print(f"[CV] Florence OCR error: {exc}")
            return "", []

    def _run_ocr(self, pil_img) -> str:
        """Legacy wrapper returning just the string."""
        text, _ = self._run_ocr_detail(pil_img, scale=1.0, offset=(0,0))
        return text

    def _multipass_ocr(self, enhanced_img, crop_img, scale: float, offset: tuple) -> tuple:
        """
        Multi-pass OCR strategy for maximum accuracy:
          Pass 1: enhanced crop (primary)
          Pass 2: grayscale crop  (picks up different textures)
        Merges unique text tokens across all passes.
        Returns (merged_raw_text, all_lines_list)
        """
        if not self.ocr_available:
            return "", []

        from PIL import ImageOps
        all_lines  = []
        seen_texts = set()
        merged     = []

        passes = [
            (enhanced_img, "enhanced", scale),
        ]
        
        # Disable the second grayscale pass on CPU to satisfy the < 2s latency constraint
        if getattr(self, "device", "cpu") != "cpu":
            from PIL import ImageOps
            passes.append((ImageOps.grayscale(crop_img).convert("RGB"), "grayscale", 1.0))

        for img, pass_name, pass_scale in passes:
            text, lines = self._run_ocr_detail(img, scale=pass_scale, offset=offset)
            for ln in lines:
                t = ln["text"].strip()
                t_key = t.lower().replace(" ","")
                if t_key and t_key not in seen_texts and len(t) > 1:
                    seen_texts.add(t_key)
                    all_lines.append({**ln, "pass": pass_name})
                    merged.append(t)

        return " | ".join(merged), all_lines

    # ── Stage 4: Improved Parser ──────────────────────────────────────────

    def _parse(self, text: str) -> dict:
        """
        Extract structured wine info from raw OCR text.
        Returns field values + per-field confidence dict.
        """
        tl  = text.lower()
        out: dict = {"confidence": 0.0, "field_confidence": {}}
        fc  = out["field_confidence"]

        # ── Year ──────────────────────────────────────────────────────────
        years = _YEAR_RE.findall(text)
        # Filter out volume (75cl = 75, not a year) and alcohol (%)
        vol_matches  = set(_VOLUME_RE.findall(text))
        alc_matches  = set(_ALCOHOL_RE.findall(text))
        years = [y for y in years if y not in vol_matches]
        if years:
            out["year"] = years[0]
            out["confidence"] += 0.20
            fc["year"] = 0.95
        else:
            fc["year"] = 0.0

        # ── Country ───────────────────────────────────────────────────────
        country_conf = 0.0
        for country, kws in COUNTRY_KEYWORDS.items():
            hits = [kw for kw in kws if kw in tl]
            if hits:
                score = min(1.0, 0.7 + len(hits) * 0.1)
                if score > country_conf:
                    out["country"] = country
                    country_conf   = score
        if out.get("country"):
            out["confidence"] += 0.25
            fc["country"] = round(country_conf, 2)
        else:
            fc["country"] = 0.0

        # ── Variety — exact then fuzzy ─────────────────────────────────────
        variety_found = False
        # Pass 1: exact substring match (fast)
        for variety in WINE_VARIETIES_SORTED:
            if variety.lower() in tl:
                out["variety"] = variety
                out["confidence"] += 0.25
                fc["variety"] = 0.95
                variety_found = True
                break
        # Pass 2: fuzzy match with rapidfuzz (handles OCR typos like 'Malbzc')
        if not variety_found:
            try:
                from rapidfuzz import process, fuzz
                # Try each OCR token individually
                tokens = [t.strip(".,|!:;()") for t in text.split() if len(t.strip()) > 4]
                for token in tokens:
                    match = process.extractOne(
                        token, WINE_VARIETIES,
                        scorer=fuzz.ratio,
                        score_cutoff=82,
                    )
                    if match:
                        out["variety"] = match[0]
                        fuzzy_conf = round(match[1] / 100, 2)
                        out["confidence"] += 0.20
                        fc["variety"] = fuzzy_conf
                        variety_found = True
                        break
            except ImportError:
                pass
        if not variety_found:
            fc["variety"] = 0.0

        # Price removed
        out["price"] = None
        fc["price"] = 0.0

        # ── Score / critic points ──────────────────────────────────────────
        sm = _SCORE_RE.findall(text)
        if sm:
            out["score"] = int(sm[0])
            fc["score"] = 0.90

        # ── Title ─────────────────────────────────────────────────────────
        skip_lower = (
            {kw for kws in COUNTRY_KEYWORDS.values() for kw in kws}
            | {v.lower() for v in WINE_VARIETIES}
            | _TITLE_SKIP
        )
        cap_tokens = [
            w.strip(".,!:;()[]\"'\u2019|")
            for w in text.replace("|", " ").split()
            if w and w[0].isupper() and len(w) > 2
            and not w.isdigit()
            and not re.match(r"^\d", w)
        ]
        name_tokens = [t for t in cap_tokens if t.lower() not in skip_lower and len(t) > 2]
        if name_tokens:
            out["title"] = " ".join(name_tokens[:6])
            out["confidence"] += 0.20
            fc["title"] = 0.70
        else:
            fc["title"] = 0.0

        out["confidence"] = round(min(out["confidence"], 1.0), 2)
        return out

    def _annotate_image(self, pil_img, all_boxes: list, best_box, all_ocr_lines: list = None) -> str:
        """
        Draw YOLO detection boxes and OCR bounding boxes onto a copy of the original image.
        - all_boxes: grey dashed borders (all detections)
        - best_box:  bright green solid border + label area highlight
        - all_ocr_lines: orange boxes for extracted text components

        Returns a base64-encoded JPEG string (for embedding in JSON/HTML).
        """
        import base64
        try:
            from PIL import ImageDraw, ImageFont

            annotated = pil_img.copy().convert("RGBA")
            overlay   = annotated.copy()
            draw_ov   = ImageDraw.Draw(overlay, "RGBA")
            draw_ann  = ImageDraw.Draw(annotated)

            # Draw all detected boxes (translucent grey)
            for b in all_boxes:
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                cls_name = b.get("cls_name", "?")
                conf     = b.get("conf", 0)
                # skip tiny boxes
                if (x2-x1) < 20 or (y2-y1) < 20:
                    continue
                draw_ov.rectangle([x1,y1,x2,y2],
                                   outline=(150,150,150,180), width=1)
                label = f"{cls_name} {conf:.2f}"
                draw_ov.text((x1+3, y1+2), label, fill=(200,200,200,200))

            # Draw best crop box (bright green + semi-transparent fill)
            if best_box:
                bx1, by1, bx2, by2 = best_box
                draw_ov.rectangle([bx1,by1,bx2,by2],
                                   fill=(0,255,120,30),
                                   outline=(0,255,120,255), width=3)
                label = f"Label Crop (conf: {best_box[4]:.2f})" if len(best_box) > 4 else "Label Crop"
                draw_ov.text((bx1+5, by1+5), label, fill=(0,255,120,230))
                draw_ann.text((bx1+4, by1+4), label, fill=(0,255,100,255))

            # Draw OCR text bounding boxes (orange/gold)
            if all_ocr_lines:
                for ln in all_ocr_lines:
                    bbox = ln.get("bbox")
                    if bbox and len(bbox) == 4:
                        # Draw a polygon (4 points) since text can be rotated
                        poly = [tuple(pt) for pt in bbox]
                        # Outline orange, slight translucent fill
                        draw_ov.polygon(poly, outline=(250,180,50,220), width=2)
                        
                        # Add text label above it with high confidence in green
                        conf = ln.get("conf", 0)
                        text_col = (50,255,100,255) if conf > 0.8 else (250,180,50,255)
                        # Draw text on annotated image so it's opaque
                        draw_ann.text((bbox[0][0], max(0, bbox[0][1]-12)), ln.get("text",""), fill=text_col)

            # Composite overlay onto annotated
            annotated = annotated.convert("RGB")
            from PIL import Image as PILImage
            annotated = PILImage.alpha_composite(
                pil_img.convert("RGBA"), overlay
            ).convert("RGB")

            # Encode to JPEG base64
            buf = io.BytesIO()
            annotated.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as exc:
            print(f"[CV] Annotate failed: {exc}")
            # Return original image as fallback
            try:
                buf = io.BytesIO()
                pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return ""

    # ── CLIP Embedding ────────────────────────────────────────────────────────

    def _clip_embed(self, pil_img) -> Optional[np.ndarray]:
        """Return L2-normalised CLIP image embedding (512-dim)."""
        try:
            import torch
            if not self.clip_available or self._clip_model is None:
                return None
            device = self._clip_model.device
            inputs = self._clip_processor(images=pil_img, return_tensors="pt")
            if device.type == "cuda":
                inputs = {k: v.to(device, dtype=torch.float16) if k == "pixel_values" else v.to(device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = self._clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].cpu().numpy()
        except Exception as exc:
            print(f"[CV] CLIP embed error: {exc}")
            return None

    def _clip_embed_text(self, text: str) -> Optional[np.ndarray]:
        """Return L2-normalised CLIP text embedding (512-dim)."""
        try:
            import torch
            if not self.clip_available or self._clip_model is None:
                return None
            device = self._clip_model.device
            inputs = self._clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = self._clip_model.get_text_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].cpu().numpy()
        except Exception as exc:
            print(f"[CV] CLIP text embed error: {exc}")
            return None

    def _clip_embed_texts(self, texts: list) -> Optional[np.ndarray]:
        """Return L2-normalised CLIP text embeddings for a list of texts (N × 512)."""
        try:
            import torch
            if not self.clip_available or self._clip_model is None:
                return None
            device = self._clip_model.device
            inputs = self._clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                feat = self._clip_model.get_text_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.cpu().numpy()
        except Exception as exc:
            print(f"[CV] CLIP batched text embed error: {exc}")
            return None

    def preprocess_cylindrical_image(self, pil_img, radius_ratio=0.85):
        """
        Perform 2D cylindrical unwarping to correct geometric curvature distortion
        on label text near the edges of a wine bottle using vectorized numpy grids.
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image as PILImage
            
            img_np = np.array(pil_img)
            h, w = img_np.shape[:2]
            
            # Cylinder radius R based on width and radius_ratio
            R = (w / 2.0) / radius_ratio
            xc = w / 2.0
            
            # Vectorized coordinate grids
            ys, xs = np.indices((h, w), dtype=np.float32)
            
            # Calculate offset from center
            dx = xs - xc
            theta = dx / R
            
            # Compute source x and y mapping
            # Mask out invalid theta range (beyond 90 degrees)
            valid_mask = (theta > -np.pi/2) & (theta < np.pi/2)
            
            map_x = np.where(valid_mask, R * np.sin(theta) + xc, -1.0).astype(np.float32)
            map_y = np.where(valid_mask, ys, -1.0).astype(np.float32)
            
            # Remap using OpenCV cubic interpolation
            unwarped = cv2.remap(img_np, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return PILImage.fromarray(unwarped)
        except Exception as exc:
            print(f"[CV] Cylindrical unwarping processing error: {exc}")
            return pil_img

    def _cylindrical_unwarp(self, pil_img, radius_ratio=0.85):
        """Legacy alias to ensure backward compatibility."""
        return self.preprocess_cylindrical_image(pil_img, radius_ratio)

    # ── Color Detection ───────────────────────────────────────────────────────

    def _detect_color(self, pil_img) -> str:
        """Simple RGB heuristic to detect red / white / rosé wine color."""
        try:
            w, h = pil_img.size
            # Sample center strip (often the wine itself in bottle photos)
            crop = pil_img.crop((w // 3, h // 3, 2 * w // 3, 2 * h // 3))
            arr  = np.array(crop).astype(float)
            r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
            if r > 130 and g < 90  and b < 90:   return "red"
            if r > 180 and g > 100 and b < 100:  return "rosé"
            if g >= r  or g >= b:                return "white"
            return "red"   # safe default for wine bottles
        except Exception:
            return "unknown"


# ─── Aroma / Tasting Note Helpers ────────────────────────────────────────────

def get_aroma_profile(variety: str) -> dict:
    """Return aroma profile dict for a given grape variety (fuzzy match)."""
    if not variety:
        return {}
    variety_lc = variety.lower()
    for key, profile in AROMA_PROFILES.items():
        if key.lower() in variety_lc or variety_lc in key.lower():
            return profile
    return {}


def get_food_pairings(variety: str) -> list:
    """Return food pairing list for a given variety."""
    if not variety:
        return []
    variety_lc = variety.lower()
    for key, pairings in FOOD_PAIRINGS.items():
        if key.lower() in variety_lc or variety_lc in key.lower():
            return pairings
    return []


def build_tasting_notes(wine: dict) -> str:
    """Generate a human-readable tasting-notes paragraph from wine metadata."""
    title   = wine.get("title", "This wine")
    variety = wine.get("variety", "")
    country = wine.get("country", "")
    price   = wine.get("price")
    desc    = wine.get("description", "")

    profile  = get_aroma_profile(variety)
    pairings = get_food_pairings(variety)

    parts = [f"**{title}**"]

    if variety and country:
        parts.append(f"is a {variety} from {country}.")
    elif variety:
        parts.append(f"is a {variety}.")

    if profile:
        primary  = ", ".join(profile.get("primary",   [])[:3])
        secondary= ", ".join(profile.get("secondary", [])[:2])
        palate   = profile.get("palate", "")
        color    = profile.get("color", "")
        if color:
            parts.append(f"Color: {color}.")
        if primary:
            parts.append(f"Aroma: notes of {primary}.")
        if secondary:
            parts.append(f"Complexity: hints of {secondary}.")
        if palate:
            parts.append(f"Palate: {palate}.")
    elif desc:
        # Fallback to description snippet
        parts.append(desc[:200] + ("…" if len(desc) > 200 else ""))

    if pairings:
        parts.append(f"Pairs well with: {', '.join(pairings[:4])}.")

    return " ".join(parts)


# ─── Catalog Search Helpers ───────────────────────────────────────────────────

def find_in_catalog(scan_info: dict, catalog_df) -> dict:
    """
    Find the scanned wine in the catalog using fuzzy title and grape variety similarity,
    avoiding strict matching of secondary fields (overfitting).
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        process = None
        fuzz = None

    scanned_title = (scan_info.get("title") or "").strip()
    scanned_variety = (scan_info.get("variety") or "").strip().lower()

    if not scanned_title:
        return {"found": False, "match_type": None, "similarity": 0.0, "wine": {}}

    # 1. Use rapidfuzz process.extract to quickly find the top 15 candidate titles (in C++)
    # This avoids Python iteration overhead over 130,000 items
    if process and fuzz:
        choices = catalog_df["title"].fillna("").tolist()
        matches = process.extract(
            scanned_title, choices,
            scorer=fuzz.token_set_ratio,
            limit=15
        )
        
        best_match = None
        best_score = 0.0
        
        for title_match, score_val, idx in matches:
            row = catalog_df.iloc[idx]
            variety_db = str(row.get("variety", "")).lower()
            
            # Variety match score
            variety_score = 70.0
            if scanned_variety and variety_db:
                variety_score = fuzz.token_set_ratio(scanned_variety, variety_db)
                
                # Check for synonyms or subsets (e.g. Bordeaux blends are Cabernet-based, Red Blend / Red Wine)
                variety_db_clean = variety_db.replace("-", " ").lower()
                scanned_variety_clean = scanned_variety.replace("-", " ").lower()
                
                if "cabernet" in scanned_variety_clean and "bordeaux" in variety_db_clean:
                    variety_score = max(variety_score, 80.0)
                if "red blend" in scanned_variety_clean and "bordeaux" in variety_db_clean:
                    variety_score = max(variety_score, 85.0)
                if "bordeaux" in scanned_variety_clean and "red blend" in variety_db_clean:
                    variety_score = max(variety_score, 85.0)
                if "red wine" in scanned_variety_clean and "red blend" in variety_db_clean:
                    variety_score = max(variety_score, 80.0)
                if "cabernet" in scanned_variety_clean and "red blend" in variety_db_clean:
                    variety_score = max(variety_score, 80.0)
                
            # If title is extremely similar (high confidence match), relax the variety requirement
            is_match = False
            if score_val >= 90:
                is_match = (variety_score >= 45)
            elif score_val >= 72:
                is_match = (variety_score >= 70)
                
            if is_match:
                combined_score = 0.7 * score_val + 0.3 * variety_score
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = row.to_dict()
                    
        if best_match:
            return {
                "found": True,
                "match_type": "fuzzy_title_variety_fast",
                "similarity": round(best_score / 100, 2),
                "wine": _clean_wine(best_match),
            }

        # 2nd Pass fallback: Space-insensitive matching to catch cases like "Opusone" -> "Opus One"
        import re
        scanned_title_clean = re.sub(r'[^a-z0-9]', '', scanned_title.lower())
        if len(scanned_title_clean) >= 4:
            if "title_clean" not in catalog_df.columns:
                catalog_df["title_clean"] = catalog_df["title"].fillna("").str.replace(r'[^a-zA-Z0-9]', '', regex=True).str.lower()
            choices_clean = catalog_df["title_clean"].tolist()
            
            matches_clean = process.extract(
                scanned_title_clean, choices_clean,
                scorer=fuzz.partial_ratio,
                limit=10
            )
            
            for title_match_clean, score_val, idx in matches_clean:
                if score_val >= 90:
                    row = catalog_df.iloc[idx]
                    variety_db = str(row.get("variety", "")).lower()
                    
                    variety_score = 70.0
                    if scanned_variety and variety_db:
                        variety_score = fuzz.token_set_ratio(scanned_variety, variety_db)
                        
                    combined_score = 0.7 * score_val + 0.3 * variety_score
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = row.to_dict()
                        
            if best_match:
                return {
                    "found": True,
                    "match_type": "fuzzy_title_variety_spaceless",
                    "similarity": round(best_score / 100, 2),
                    "wine": _clean_wine(best_match),
                }
    else:
        # Simple fallback loop if rapidfuzz not available (should not happen in this environment)
        # Check first 500 items to avoid timeout
        for idx, row in catalog_df.head(500).iterrows():
            title_db = str(row.get("title", "")).lower()
            if scanned_title.lower() in title_db:
                return {
                    "found": True,
                    "match_type": "fallback_partial",
                    "similarity": 0.75,
                    "wine": _clean_wine(row.to_dict()),
                }

    return {"found": False, "match_type": None, "similarity": 0.0, "wine": {}}


def find_similar_wines(scan_info: dict, catalog_df, n: int = 5, occasion: str = None, food_pairing: str = None, scanner = None) -> list[dict]:
    """
    Find similar wines when the scanned bottle is NOT in stock/catalog.
    Uses multimodal visual-semantic similarity (CLIP) and user context (occasion/food).
    Optimized to run in <400ms by batch-encoding CLIP text features.
    """
    variety = (scan_info.get("variety") or "").strip()
    country = (scan_info.get("country") or "").strip()
    year    = (scan_info.get("year") or "").strip()
    visual_emb = scan_info.get("visual_embedding")

    # 1. Filter candidates pool: same variety or same country to narrow search space
    pool = pd.DataFrame()
    if variety:
        m = catalog_df["variety"].str.contains(variety, case=False, na=False)
        pool = catalog_df[m].copy()
    if pool.empty and country:
        m = catalog_df["country"].str.contains(country, case=False, na=False)
        pool = catalog_df[m].copy()
    if pool.empty:
        pool = catalog_df.sample(min(100, len(catalog_df)), random_state=42).copy()

    # Limit pool size to speed up CLIP text encoding on CPU (reduce from 40 to 20 for CPU speed)
    pool = pool.head(20).copy()
    
    # 2. CLIP similarity calculation (Image-to-Image or Image-to-Text via GPU Matrix Multiplication)
    vis_scores = {}
    if visual_emb is not None and scanner is not None and scanner.clip_available:
        try:
            import torch
            device = "cuda" if check_cuda_working() else "cpu"
            q_tensor = torch.tensor(visual_emb, device=device)
            if device == "cuda":
                q_tensor = q_tensor.to(torch.float16)
                
            # If pre-computed catalog embeddings match catalog_df length, perform instant whole-catalog search
            if getattr(scanner, "catalog_embeds", None) is not None and len(scanner.catalog_embeds) == len(catalog_df):
                print("[CV] Instantly computing similarity over the entire catalog on GPU via torch.matmul...")
                sims = torch.matmul(q_tensor, scanner.catalog_embeds.T)
                sims_np = sims.cpu().numpy()
                for idx, row in pool.iterrows():
                    try:
                        catalog_idx = catalog_df.index.get_loc(idx)
                        vis_scores[idx] = max(0.0, float(sims_np[catalog_idx])) * 0.5
                    except KeyError:
                        vis_scores[idx] = 0.0
            else:
                has_candidate_images = False
                image_col = None
                for col in ["image_path", "image_file", "image_url"]:
                    if col in pool.columns:
                        image_col = col
                        break
                        
                candidate_images = []
                indices = []
                if image_col:
                    for idx, row in pool.iterrows():
                        path = str(row[image_col]).strip()
                        if path and os.path.exists(path):
                            candidate_images.append(path)
                            indices.append(idx)
                    if len(candidate_images) > 0:
                        has_candidate_images = True
                        
                if has_candidate_images:
                    # --- Image-to-Image GPU Similarity ---
                    print(f"[CV] Performing Image-to-Image similarity using CLIP Visual Encoder for {len(candidate_images)} candidates on GPU...")
                    from PIL import Image as PILImage
                    cand_embs = []
                    for path in candidate_images:
                        with PILImage.open(path).convert("RGB") as c_img:
                            c_emb = scanner._clip_embed(c_img)
                            if c_emb is not None:
                                cand_embs.append(c_emb)
                            else:
                                cand_embs.append(np.zeros(512))
                    cand_embs_t = torch.tensor(np.array(cand_embs), device=device)
                    if device == "cuda":
                        cand_embs_t = cand_embs_t.to(torch.float16)
                        
                    sims = torch.matmul(q_tensor, cand_embs_t.T)
                    sims_np = sims.cpu().numpy()
                    for idx_in_cand, orig_idx in enumerate(indices):
                        vis_scores[orig_idx] = max(0.0, float(sims_np[idx_in_cand])) * 0.5
                else:
                    # --- Image-to-Text GPU Similarity ---
                    print("[CV] Performing Image-to-Text similarity using CLIP cross-modal matching on GPU...")
                    cand_texts = []
                    indices = []
                    for idx, row in pool.iterrows():
                        cand_text = f"A wine label showing {row.get('title')}. It is a {row.get('variety')} from {row.get('country')}. Tasting notes: {row.get('description')[:120]}."
                        cand_texts.append(cand_text)
                        indices.append(idx)
                        
                    batch_embs = scanner._clip_embed_texts(cand_texts)
                    if batch_embs is not None:
                        batch_embs_t = torch.tensor(batch_embs, device=device)
                        if device == "cuda":
                            batch_embs_t = batch_embs_t.to(torch.float16)
                        sims = torch.matmul(q_tensor, batch_embs_t.T)
                        sims_np = sims.cpu().numpy()
                        for i, idx in enumerate(indices):
                            vis_scores[idx] = max(0.0, float(sims_np[i])) * 0.5
        except Exception as e:
            print(f"[WARN] CLIP GPU similarity failed: {e}. Falling back...")
                
    # 3. Score candidates
    scored_candidates = []
    
    occ_keywords = {
        "romantic": ["romantic", "candle", "anniversary", "valentine", "date"],
        "bbq": ["bbq", "barbecue", "grill", "smoke", "outdoor", "charcoal"],
        "celebration": ["celebrate", "party", "toast", "anniversary", "gift", "wedding"],
        "gift": ["gift", "present", "elegant", "box", "prestige"],
        "casual": ["casual", "daily", "simple", "easy", "friend", "weekday"],
    }
    
    food_keywords = {
        "beef": ["beef", "steak", "meat", "ribeye", "roast"],
        "steak": ["steak", "beef", "ribeye", "meat"],
        "seafood": ["seafood", "fish", "shrimp", "lobster", "crab", "oyster", "salmon"],
        "fish": ["fish", "salmon", "tuna", "seafood", "cod"],
        "cheese": ["cheese", "brie", "cheddar", "platter", "charcuterie"],
        "chicken": ["chicken", "poultry", "bird", "turkey"],
        "pasta": ["pasta", "spaghetti", "tomato", "sauce"],
    }

    occ_clean = (occasion or "").lower().strip()
    food_clean = (food_pairing or "").lower().strip()

    for idx, row in pool.iterrows():
        title_db = str(row.get("title", ""))
        desc_db = str(row.get("description", "")).lower()
        var_db = str(row.get("variety", "")).lower()
        cnt_db = str(row.get("country", "")).lower()
        
        # Meta score
        s_meta = 0.0
        if variety and variety.lower() in var_db:
            s_meta += 0.4
        if country and country.lower() in cnt_db:
            s_meta += 0.3
        if year and year in title_db:
            s_meta += 0.1
            
        # Occasion score
        s_occ = 0.0
        if occ_clean:
            if occ_clean in desc_db:
                s_occ += 0.3
            else:
                for group, kws in occ_keywords.items():
                    if group in occ_clean or occ_clean in group:
                        if any(kw in desc_db for kw in kws):
                            s_occ += 0.2
                            break
                            
        # Food pairing score
        s_food = 0.0
        if food_clean:
            if food_clean in desc_db:
                s_food += 0.4
            else:
                for group, kws in food_keywords.items():
                    if group in food_clean or food_clean in group:
                        if any(kw in desc_db for kw in kws):
                            s_food += 0.3
                            break
                            
        # Visual CLIP similarity score (computed beforehand using FAISS/numpy batch)
        s_vis = vis_scores.get(idx, 0.0)

        # Combined final score
        final_score = s_meta + s_occ + s_food + s_vis
        
        # Build sommelier match reason in English
        reasons = []
        if variety and variety.lower() in var_db:
            reasons.append(f"shares the **{row.get('variety')}** grape variety")
        if country and country.lower() in cnt_db:
            reasons.append(f"comes from **{row.get('country')}**")
            
        context_reasons = []
        if occ_clean:
            context_reasons.append(f"suits your **{occasion}** setting")
        if food_clean:
            context_reasons.append(f"complements **{food_pairing}** perfectly")
            
        reason_str = "This wine "
        if reasons:
            reason_str += ", and ".join(reasons)
        else:
            reason_str += "is a wonderful alternative"
            
        if context_reasons:
            reason_str += ". It " + " and ".join(context_reasons)
            
        reason_str += "."

        w_dict = _clean_wine(row.to_dict())
        w_dict["_score"] = final_score
        w_dict["match_reason"] = reason_str
        scored_candidates.append(w_dict)

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x["_score"], reverse=True)
    
    # Return top N
    out = []
    for w in scored_candidates[:n]:
        # remove internal score tracker
        del w["_score"]
        out.append(w)
    return out


# ─── Utilities ────────────────────────────────────────────────────────────────

def _clean_wine(wine: dict) -> dict:
    """Normalise a wine dict for JSON serialisation."""
    cleaned = {}
    for k, v in wine.items():
        if isinstance(v, float) and (v != v):   # NaN
            cleaned[k] = None
        elif hasattr(v, "item"):                 # numpy scalar
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    # Ensure price is float or None
    try:
        cleaned["price"] = float(cleaned["price"]) if cleaned.get("price") else None
    except (ValueError, TypeError):
        cleaned["price"] = None
    return cleaned


# ─── CLI benchmark ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  cv_wine.py — parser smoke test (no image needed)")
    print("=" * 60)
    scanner = WineLabelScanner()

    fake_labels = [
        "Jordan 2016 Cabernet Sauvignon Alexander Valley | California, USA | $47",
        "Château Margaux 2015 Bordeaux | France | 98 pts",
        "Clos de los Siete 2018 Malbec | Mendoza Argentina",
        "Kim Crawford Sauvignon Blanc 2022 | Marlborough New Zealand",
        "Barolo DOCG 2017 Nebbiolo | Piemonte Italy | $65",
    ]
    for label in fake_labels:
        res = scanner._parse(label)
        print(f"\nText  : {label[:70]}")
        print(f"Parsed: variety={res.get('variety')}  country={res.get('country')}  "
              f"year={res.get('year')}  price={res.get('price')}  "
              f"conf={res.get('confidence', 0):.2f}")
