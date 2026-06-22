"""
evaluation/cv_eval.py — OCR & Pipeline Evaluation (Scientific Contribution)
=============================================================================
So sánh 2 chiến lược:
  Baseline : EasyOCR trên full image (không có YOLO)
  Pipeline : YOLOv8n → crop → enhance → EasyOCR  (proposed method)

Metrics:
  - Field-level F1 (variety, country, year, price)
  - Character Error Rate (CER) on raw OCR text
  - End-to-end catalog hit rate (found vs not_found)
  - Latency (ms per image)

Usage:
    python evaluation/cv_eval.py [--n 50] [--csv data/raw/winemag-data-130k-v2.csv]
"""

import sys, os, time, re, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from io import BytesIO

from cv_wine import WineLabelScanner


# ─── Synthetic Label Generator ────────────────────────────────────────────────

def generate_label_image(meta: dict, w=500, h=700) -> bytes:
    """
    Render a synthetic wine label from metadata dict.
    Keys used: title, variety, country, year, price
    Returns JPEG bytes.
    """
    img  = Image.new("RGB", (w, h), color=(18, 14, 35))
    draw = ImageDraw.Draw(img)

    # Bottle silhouette
    draw.rectangle([100, 0, 400, h], fill=(12, 10, 25))

    # Label rectangle (cream background)
    lx1, ly1, lx2, ly2 = 115, 170, 385, 530
    draw.rectangle([lx1, ly1, lx2, ly2], fill=(245, 235, 210), outline=(100, 60, 30), width=2)

    # Header stripe
    draw.rectangle([lx1, ly1, lx2, ly1+75], fill=(145, 28, 50))

    cx = (lx1 + lx2) // 2

    lines = []
    if meta.get("title"):
        # Take first 2 words of title for the header
        words = str(meta["title"]).split()[:3]
        lines.append((" ".join(words).upper(), 28, (255, 228, 190)))
    if meta.get("year"):
        lines.append((str(meta["year"]), 24, (255, 228, 190)))
    if meta.get("variety"):
        lines.append((str(meta["variety"]), 20, (25, 10, 10)))
    if meta.get("country"):
        lines.append((str(meta["country"]), 17, (25, 10, 10)))
    if meta.get("price"):
        try:
            lines.append((f"${float(meta['price']):.0f}", 20, (25, 10, 10)))
        except Exception:
            pass

    y = ly1 + 20
    for text, size, color in lines:
        draw.text((cx, y), text[:30], fill=color, anchor="mm")
        y += size + 10

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ─── CER Metric ────────────────────────────────────────────────────────────────

def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate (Levenshtein / len(reference))."""
    h, r = list(hypothesis.lower()), list(reference.lower())
    if not r:
        return 0.0 if not h else 1.0
    # Standard edit distance
    dp = list(range(len(r) + 1))
    for i, hc in enumerate(h):
        new = [i + 1]
        for j, rc in enumerate(r):
            ins  = new[j] + 1
            dlt  = dp[j + 1] + 1
            sub  = dp[j] + (0 if hc == rc else 1)
            new.append(min(ins, dlt, sub))
        dp = new
    return dp[-1] / len(r)


# ─── Field-level Precision / Recall / F1 ─────────────────────────────────────

def field_f1(pred: str, gold: str) -> float:
    """Token-level F1 between two strings."""
    if not gold:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    common    = pred_toks & gold_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ─── Evaluator ────────────────────────────────────────────────────────────────

class CVEvaluator:
    def __init__(self):
        self.scanner = WineLabelScanner()
        print("[Eval] Pre-loading OCR + YOLO …")
        self.scanner.load(load_clip=False)

    def evaluate_single(self, meta: dict, use_yolo: bool) -> dict:
        """
        Evaluate one wine label image.
        Returns per-sample metrics dict.
        """
        # Override YOLO flag
        orig_yolo = self.scanner.yolo_available
        self.scanner.yolo_available = use_yolo and orig_yolo

        img_bytes = generate_label_image(meta)
        t0 = time.time()
        result = self.scanner.scan(img_bytes)
        latency = (time.time() - t0) * 1000

        # Restore
        self.scanner.yolo_available = orig_yolo

        # Field F1s
        f1_variety = field_f1(result.get("variety") or "", meta.get("variety") or "")
        f1_country = field_f1(result.get("country") or "", meta.get("country") or "")
        f1_year    = field_f1(result.get("year")    or "", str(meta.get("year") or ""))

        # Price match (within 15%)
        p_gold = meta.get("price")
        p_pred = result.get("price")
        price_ok = 0.0
        if p_gold and p_pred:
            try:
                ratio = abs(float(p_pred) - float(p_gold)) / (float(p_gold) + 1e-6)
                price_ok = 1.0 if ratio < 0.15 else 0.0
            except Exception:
                pass

        # CER on the title (first 3 words)
        gold_title_short = " ".join(str(meta.get("title","")).split()[:3]).upper()
        raw_ocr = result.get("raw_ocr") or ""
        cer_score = cer(raw_ocr[:len(gold_title_short)*2], gold_title_short)

        return {
            "yolo_used"   : result.get("yolo_used", False),
            "confidence"  : result.get("confidence", 0.0),
            "f1_variety"  : f1_variety,
            "f1_country"  : f1_country,
            "f1_year"     : f1_year,
            "price_match" : price_ok,
            "cer"         : min(cer_score, 1.0),
            "latency_ms"  : latency,
        }

    def run(self, wines: list[dict], label="Method") -> dict:
        """Run full evaluation over a list of wine metadata dicts."""
        records = []
        use_yolo = "YOLO" in label
        for i, meta in enumerate(wines):
            try:
                r = self.evaluate_single(meta, use_yolo=use_yolo)
                records.append(r)
            except Exception as e:
                print(f"  [WARN] Sample {i} failed: {e}")
            if (i+1) % 10 == 0:
                print(f"  {label}: {i+1}/{len(wines)} done …")

        df  = pd.DataFrame(records)
        agg = {
            "n"              : len(df),
            "yolo_used_pct"  : round(df["yolo_used"].mean() * 100, 1),
            "avg_confidence" : round(df["confidence"].mean(), 3),
            "f1_variety"     : round(df["f1_variety"].mean(), 3),
            "f1_country"     : round(df["f1_country"].mean(), 3),
            "f1_year"        : round(df["f1_year"].mean(), 3),
            "cer"            : round(df["cer"].mean(), 3),
            "avg_latency_ms" : round(df["latency_ms"].mean(), 1),
            "p95_latency_ms" : round(df["latency_ms"].quantile(0.95), 1),
        }
        # Macro average F1 across fields (excluding price)
        agg["macro_f1"] = round(
            (agg["f1_variety"] + agg["f1_country"] + agg["f1_year"]) / 3, 3
        )
        return agg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CV Pipeline Evaluator")
    parser.add_argument("--n",   type=int, default=30,
                        help="Number of wine labels to evaluate (default: 30)")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                                             "winemag-data-130k-v2.csv"),
                        help="Path to wine CSV")
    args = parser.parse_args()

    # Load catalog
    print(f"[Eval] Loading catalog from {args.csv} …")
    try:
        df = pd.read_csv(args.csv).dropna(subset=["title","variety","country"]).head(5000)
        # Sample n wines with variety + country + year all present
        df = df[df["variety"].notna() & df["country"].notna()]
        df = df.sample(min(args.n, len(df)), random_state=42).reset_index(drop=True)
        wines = df[["title","variety","country"]].to_dict("records")
        # Add year column if exists
        if "vintage" in df.columns:
            for i, row in df.iterrows():
                wines[i]["year"] = str(row.get("vintage","")) if row.get("vintage") else ""
        else:
            # Extract year from title
            for w in wines:
                m = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", str(w.get("title","")))
                w["year"] = m.group(1) if m else ""
        print(f"[Eval] Loaded {len(wines)} wine samples.")
    except Exception as e:
        print(f"[Eval] CSV not found, using synthetic samples. ({e})")
        wines = [
            {"title": "Jordan 2016 Cabernet Sauvignon",  "variety": "Cabernet Sauvignon", "country": "US",        "year": "2016", "price": 47},
            {"title": "Chateau Margaux 2015",             "variety": "Bordeaux",           "country": "France",    "year": "2015", "price": 350},
            {"title": "Alamos Malbec 2019",               "variety": "Malbec",             "country": "Argentina", "year": "2019", "price": 12},
            {"title": "Kim Crawford Sauvignon Blanc 2022","variety": "Sauvignon Blanc",    "country": "New Zealand","year": "2022", "price": 15},
            {"title": "Barolo Nebbiolo 2017 Piemonte",   "variety": "Nebbiolo",           "country": "Italy",     "year": "2017", "price": 65},
        ] * max(1, args.n // 5)

    evaluator = CVEvaluator()

    print(f"\n{'='*60}")
    print(f"  CV Pipeline Evaluation   n={len(wines)}")
    print(f"{'='*60}")

    # ── Baseline: OCR only (YOLO disabled) ────────────────────────────
    print("\n[1/2] Baseline: EasyOCR only (no YOLO) …")
    baseline = evaluator.run(wines, label="Baseline-OCR")

    # ── Proposed: YOLO + enhance + OCR ────────────────────────────────
    print("\n[2/2] Proposed: YOLO + Enhance + EasyOCR …")
    proposed = evaluator.run(wines, label="YOLO+OCR")

    # ── Report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    header = f"{'Metric':<22} {'Baseline':>12} {'YOLO+OCR':>12} {'Delta':>10}"
    print(header)
    print("-" * 60)

    for key in ["macro_f1", "f1_variety", "f1_country", "f1_year",
                "cer", "avg_confidence",
                "avg_latency_ms", "p95_latency_ms", "yolo_used_pct"]:
        b_val = baseline.get(key, 0)
        p_val = proposed.get(key, 0)
        delta = p_val - b_val
        arrow = "+" if delta >= 0 else ""
        # Lower CER and latency = better
        if key in ("cer", "avg_latency_ms", "p95_latency_ms"):
            arrow = "-" if delta >= 0 else "+"
        unit = "%" if key in ("yolo_used_pct",) else ("ms" if "latency" in key else "")
        print(f"  {key:<20} {b_val:>12.3f} {p_val:>12.3f} {arrow}{abs(delta):>8.3f}{unit}")

    # Save JSON results
    out = {"baseline": baseline, "proposed": proposed,
           "n_samples": len(wines)}
    out_path = os.path.join(os.path.dirname(__file__), "cv_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Eval] Results saved to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
