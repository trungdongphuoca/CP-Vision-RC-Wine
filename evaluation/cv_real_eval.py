"""
evaluation/cv_real_eval.py — Real Image CV Evaluation Framework
================================================================
Evaluate the WineLabelScanner pipeline on real (or synthetic-fallback)
wine label images against per-sample JSON ground truth.

Ground Truth Format (data/cv_ground_truth/):
    image_001.jpg  — the photo
    image_001.json — {"title": "...", "variety": "...", "country": "...",
                       "year": "...", "price": ...}

Metrics computed:
    1. Field-level Token F1 for title, variety, country
    2. Year Exact Match accuracy
    3. Price Match (within 15% tolerance)
    4. CER (Character Error Rate) via Levenshtein distance
    5. Macro F1 — average across all fields
    6. Overall Detection Rate — % of images with ANY field extracted
    7. Latency — avg ms per image + P95

Usage:
    python evaluation/cv_real_eval.py \\
        --data_dir data/cv_ground_truth/ \\
        --output   evaluation/cv_real_eval_results.json

Authors: Wine AI Team
"""

import sys
import os
import json
import time
import glob
import argparse
import datetime
from pathlib import Path
from io import BytesIO
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# ─── Levenshtein Distance (pure Python, no external deps) ────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate = Levenshtein(hyp, ref) / len(ref).

    Returns 0.0 when both are empty; 1.0 when ref is empty but hyp is not.
    Capped at 1.0 for cleaner reporting.
    """
    h = hypothesis.lower().strip()
    r = reference.lower().strip()
    if not r:
        return 0.0 if not h else 1.0
    return min(_levenshtein(h, r) / len(r), 1.0)


# ─── Token F1 ─────────────────────────────────────────────────────────────────

def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference strings.

    Tokens are lowercased whitespace-split words.
    Returns 1.0 if both are empty; 0.0 if one is empty and the other is not.
    """
    if not reference:
        return 1.0 if not prediction else 0.0
    if not prediction:
        return 0.0

    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    common = pred_tokens & ref_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


# ─── Year Exact Match ────────────────────────────────────────────────────────

def year_exact_match(pred_year: Optional[str], gt_year: Optional[str]) -> float:
    """Return 1.0 if predicted year matches ground truth exactly, else 0.0."""
    if not gt_year:
        return 1.0 if not pred_year else 0.0
    return 1.0 if str(pred_year).strip() == str(gt_year).strip() else 0.0


# ─── Price Match (within tolerance) ──────────────────────────────────────────

def price_match(pred_price, gt_price, tolerance: float = 0.15) -> float:
    """
    Return 1.0 if predicted price is within `tolerance` (15%) of ground truth.
    Returns 0.0 if either price is missing or cannot be parsed.
    """
    if gt_price is None:
        return 1.0 if pred_price is None else 0.0
    if pred_price is None:
        return 0.0
    try:
        p = float(pred_price)
        g = float(gt_price)
        if g == 0:
            return 1.0 if p == 0 else 0.0
        return 1.0 if abs(p - g) / g <= tolerance else 0.0
    except (ValueError, TypeError):
        return 0.0


# ─── Synthetic Image Generator (fallback) ────────────────────────────────────

def generate_synthetic_image(meta: dict, width: int = 500, height: int = 700) -> bytes:
    """
    Generate a synthetic wine label image from ground truth metadata.
    Used as fallback when real image file does not exist.

    Parameters
    ----------
    meta : dict
        Ground truth dict with keys: title, variety, country, year, price.
    width, height : int
        Canvas dimensions in pixels.

    Returns
    -------
    bytes
        JPEG-encoded image bytes.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), color=(18, 14, 35))
    draw = ImageDraw.Draw(img)

    # Bottle silhouette
    draw.rectangle([100, 0, 400, height], fill=(12, 10, 25))

    # Label rectangle (cream background)
    lx1, ly1, lx2, ly2 = 115, 170, 385, 530
    draw.rectangle([lx1, ly1, lx2, ly2],
                   fill=(245, 235, 210), outline=(100, 60, 30), width=2)

    # Header stripe (wine-red)
    draw.rectangle([lx1, ly1, lx2, ly1 + 75], fill=(145, 28, 50))

    cx = (lx1 + lx2) // 2

    # Text lines to render
    lines = []
    if meta.get("title"):
        words = str(meta["title"]).split()[:3]
        lines.append((" ".join(words).upper(), 28, (255, 228, 190)))
    if meta.get("year"):
        lines.append((str(meta["year"]), 24, (255, 228, 190)))
    if meta.get("variety"):
        lines.append((str(meta["variety"]), 20, (25, 10, 10)))
    if meta.get("country"):
        lines.append((str(meta["country"]), 17, (25, 10, 10)))
    if meta.get("price") is not None:
        try:
            lines.append((f"${float(meta['price']):.0f}", 20, (25, 10, 10)))
        except (ValueError, TypeError):
            pass

    y = ly1 + 20
    for text, size, color in lines:
        draw.text((cx, y), text[:30], fill=color, anchor="mm")
        y += size + 10

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ─── Dataset Loader ──────────────────────────────────────────────────────────

def load_ground_truth_dataset(data_dir: str) -> list[dict]:
    """
    Load paired (image, ground_truth) samples from a directory.

    Expects files named:
        image_NNN.json  — ground truth
        image_NNN.jpg   — photo (optional; synthetic generated if missing)

    Returns list of dicts with keys:
        image_bytes, ground_truth, image_id, is_synthetic
    """
    data_dir = Path(data_dir)
    json_files = sorted(glob.glob(str(data_dir / "*.json")))

    if not json_files:
        print(f"[cv_real_eval] WARNING: No JSON files found in {data_dir}")
        print("[cv_real_eval] Generating 3 synthetic samples for demonstration...")
        return _generate_demo_dataset(data_dir)

    samples = []
    for json_path in json_files:
        json_path = Path(json_path)
        stem = json_path.stem  # e.g. "image_001"

        # Load ground truth
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        # Try to find corresponding image
        img_bytes = None
        is_synthetic = True
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            img_path = json_path.with_suffix(ext)
            if img_path.exists():
                with open(img_path, "rb") as f_img:
                    img_bytes = f_img.read()
                is_synthetic = False
                break

        # Fallback: generate synthetic image from ground truth
        if img_bytes is None:
            img_bytes = generate_synthetic_image(gt)
            is_synthetic = True

        samples.append({
            "image_bytes": img_bytes,
            "ground_truth": gt,
            "image_id": stem,
            "is_synthetic": is_synthetic,
        })

    print(f"[cv_real_eval] Loaded {len(samples)} samples from {data_dir}")
    n_real = sum(1 for s in samples if not s["is_synthetic"])
    n_synth = sum(1 for s in samples if s["is_synthetic"])
    print(f"[cv_real_eval]   Real images: {n_real}, Synthetic fallback: {n_synth}")
    return samples


def _generate_demo_dataset(data_dir: Path) -> list[dict]:
    """Generate 3 demo samples when no data is present."""
    demos = [
        {
            "title": "Jordan Estate Cabernet Sauvignon Alexander Valley",
            "variety": "Cabernet Sauvignon",
            "country": "US",
            "year": "2016",
            "price": 47.0,
        },
        {
            "title": "Château Margaux Grand Vin Bordeaux",
            "variety": "Bordeaux",
            "country": "France",
            "year": "2015",
            "price": 350.0,
        },
        {
            "title": "Alamos Selección Malbec Mendoza",
            "variety": "Malbec",
            "country": "Argentina",
            "year": "2019",
            "price": 12.0,
        },
    ]
    samples = []
    for i, gt in enumerate(demos, 1):
        samples.append({
            "image_bytes": generate_synthetic_image(gt),
            "ground_truth": gt,
            "image_id": f"demo_{i:03d}",
            "is_synthetic": True,
        })
    return samples


# ─── Single-Sample Evaluation ────────────────────────────────────────────────

def evaluate_sample(scanner, sample: dict) -> dict:
    """
    Run the scanner on one sample and compute all metrics.

    Parameters
    ----------
    scanner : WineLabelScanner
        Initialized scanner instance.
    sample : dict
        Dict with keys: image_bytes, ground_truth, image_id, is_synthetic.

    Returns
    -------
    dict
        Per-sample metrics including field F1s, CER, price_match, etc.
    """
    gt = sample["ground_truth"]
    img_bytes = sample["image_bytes"]

    # Run pipeline with timing
    t0 = time.perf_counter()
    result = scanner.scan(img_bytes)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Extract predictions
    pred_title = result.get("title") or ""
    pred_variety = result.get("variety") or ""
    pred_country = result.get("country") or ""
    pred_year = result.get("year") or ""
    pred_price = result.get("price")

    # Extract ground truth
    gt_title = gt.get("title") or ""
    gt_variety = gt.get("variety") or ""
    gt_country = gt.get("country") or ""
    gt_year = gt.get("year") or ""
    gt_price = gt.get("price")

    # ── Compute metrics ──────────────────────────────────────────────
    f1_title = token_f1(pred_title, gt_title)
    f1_variety = token_f1(pred_variety, gt_variety)
    f1_country = token_f1(pred_country, gt_country)
    year_em = year_exact_match(pred_year, gt_year)
    pm = price_match(pred_price, gt_price, tolerance=0.15)

    # CER on raw OCR vs concatenated ground truth
    gt_concat = " ".join(filter(None, [gt_title, gt_variety, gt_country, gt_year]))
    raw_ocr = result.get("raw_ocr") or ""
    cer_score = cer(raw_ocr, gt_concat)

    # Detection: any field was extracted
    any_detected = any([pred_title, pred_variety, pred_country, pred_year,
                        pred_price is not None])

    return {
        "image_id": sample["image_id"],
        "is_synthetic": sample["is_synthetic"],
        # Predictions
        "pred_title": pred_title,
        "pred_variety": pred_variety,
        "pred_country": pred_country,
        "pred_year": pred_year,
        "pred_price": pred_price,
        # Ground truth
        "gt_title": gt_title,
        "gt_variety": gt_variety,
        "gt_country": gt_country,
        "gt_year": gt_year,
        "gt_price": gt_price,
        # Metrics
        "f1_title": round(f1_title, 4),
        "f1_variety": round(f1_variety, 4),
        "f1_country": round(f1_country, 4),
        "year_exact_match": round(year_em, 4),
        "price_match": round(pm, 4),
        "cer": round(cer_score, 4),
        "any_detected": any_detected,
        "latency_ms": round(latency_ms, 1),
        "confidence": result.get("confidence", 0.0),
        "yolo_used": result.get("yolo_used", False),
    }


# ─── Aggregate Metrics ──────────────────────────────────────────────────────

def aggregate_metrics(per_sample: list[dict]) -> dict:
    """
    Compute aggregate metrics from per-sample results.

    Returns
    -------
    dict
        Aggregated metrics including macro_f1, detection_rate, latency stats.
    """
    if not per_sample:
        return {"error": "No samples evaluated"}

    n = len(per_sample)

    # Field-level averages
    f1_title = np.mean([s["f1_title"] for s in per_sample])
    f1_variety = np.mean([s["f1_variety"] for s in per_sample])
    f1_country = np.mean([s["f1_country"] for s in per_sample])
    year_em = np.mean([s["year_exact_match"] for s in per_sample])
    pm = np.mean([s["price_match"] for s in per_sample])
    cer_avg = np.mean([s["cer"] for s in per_sample])

    # Macro F1: average across all field-level metrics
    macro_f1 = np.mean([f1_title, f1_variety, f1_country, year_em, pm])

    # Detection rate
    detection_rate = np.mean([1.0 if s["any_detected"] else 0.0
                              for s in per_sample])

    # Latency stats
    latencies = [s["latency_ms"] for s in per_sample]
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    max_latency = np.max(latencies)

    # YOLO usage
    yolo_pct = np.mean([1.0 if s["yolo_used"] else 0.0
                         for s in per_sample]) * 100

    # Confidence
    avg_conf = np.mean([s["confidence"] for s in per_sample])

    return {
        "n_samples": n,
        "n_real_images": sum(1 for s in per_sample if not s["is_synthetic"]),
        "n_synthetic": sum(1 for s in per_sample if s["is_synthetic"]),
        # Field-level metrics
        "f1_title": round(f1_title, 4),
        "f1_variety": round(f1_variety, 4),
        "f1_country": round(f1_country, 4),
        "year_exact_match": round(year_em, 4),
        "price_match": round(pm, 4),
        "cer": round(cer_avg, 4),
        # Aggregate
        "macro_f1": round(macro_f1, 4),
        "detection_rate": round(detection_rate, 4),
        # Latency
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(p50_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "max_latency_ms": round(max_latency, 1),
        # Pipeline info
        "yolo_used_pct": round(yolo_pct, 1),
        "avg_confidence": round(avg_conf, 4),
    }


# ─── Report Generators ──────────────────────────────────────────────────────

def generate_markdown_report(agg: dict, per_sample: list[dict],
                             output_path: str) -> str:
    """
    Generate a markdown summary table and save to file.

    Returns the markdown string.
    """
    lines = [
        "# CV Pipeline — Real Image Evaluation Report",
        "",
        f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Samples**: {agg['n_samples']} "
        f"(Real: {agg['n_real_images']}, Synthetic: {agg['n_synthetic']})",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Macro F1** | {agg['macro_f1']:.4f} |",
        f"| Title Token F1 | {agg['f1_title']:.4f} |",
        f"| Variety Token F1 | {agg['f1_variety']:.4f} |",
        f"| Country Token F1 | {agg['f1_country']:.4f} |",
        f"| Year Exact Match | {agg['year_exact_match']:.4f} |",
        f"| Price Match (±15%) | {agg['price_match']:.4f} |",
        f"| CER (lower=better) | {agg['cer']:.4f} |",
        f"| Detection Rate | {agg['detection_rate']:.4f} |",
        "",
        "## Latency",
        "",
        "| Stat | ms |",
        "|------|-----|",
        f"| Average | {agg['avg_latency_ms']:.1f} |",
        f"| P50 (Median) | {agg['p50_latency_ms']:.1f} |",
        f"| P95 | {agg['p95_latency_ms']:.1f} |",
        f"| Max | {agg['max_latency_ms']:.1f} |",
        "",
        "## Pipeline Info",
        "",
        f"- YOLO detection used: {agg['yolo_used_pct']:.1f}%",
        f"- Average confidence: {agg['avg_confidence']:.4f}",
        "",
        "## Per-Sample Results",
        "",
        "| Image | Title F1 | Variety F1 | Country F1 | Year EM | Price | CER | Latency (ms) |",
        "|-------|----------|-----------|------------|---------|-------|-----|---------------|",
    ]

    for s in per_sample:
        synth_tag = " (S)" if s["is_synthetic"] else ""
        lines.append(
            f"| {s['image_id']}{synth_tag} "
            f"| {s['f1_title']:.2f} "
            f"| {s['f1_variety']:.2f} "
            f"| {s['f1_country']:.2f} "
            f"| {s['year_exact_match']:.0f} "
            f"| {s['price_match']:.0f} "
            f"| {s['cer']:.3f} "
            f"| {s['latency_ms']:.0f} |"
        )

    lines.extend(["", f"*(S) = Synthetic image (no real photo available)*", ""])

    md = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[cv_real_eval] Markdown report saved: {output_path}")
    return md


def save_json_results(agg: dict, per_sample: list[dict],
                      output_path: str) -> None:
    """Save full results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove image_bytes-related large fields from per_sample for JSON
    clean_samples = []
    for s in per_sample:
        clean = {k: v for k, v in s.items()
                 if k not in ("image_bytes",)}
        clean_samples.append(clean)

    results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluator": "cv_real_eval.py",
            "version": "1.0.0",
        },
        "aggregate": agg,
        "per_sample": clean_samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[cv_real_eval] JSON results saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CV Real Image Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/cv_real_eval.py --data_dir data/cv_ground_truth/
  python evaluation/cv_real_eval.py --data_dir data/cv_ground_truth/ --output results.json
        """,
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(ROOT / "data" / "cv_ground_truth"),
        help="Directory containing image_NNN.{jpg,json} pairs",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(ROOT / "evaluation" / "cv_real_eval_results.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--report", type=str,
        default=str(ROOT / "evaluation" / "cv_real_eval_report.md"),
        help="Output markdown report path",
    )
    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────
    print("=" * 65)
    print("  CV Pipeline — Real Image Evaluation")
    print("=" * 65)

    # ── Load dataset ──────────────────────────────────────────────────
    samples = load_ground_truth_dataset(args.data_dir)
    if not samples:
        print("[cv_real_eval] ERROR: No samples to evaluate. Exiting.")
        sys.exit(1)

    # ── Initialize scanner ────────────────────────────────────────────
    from cv_wine import WineLabelScanner

    print("\n[cv_real_eval] Initializing WineLabelScanner...")
    scanner = WineLabelScanner()
    scanner.load(load_clip=False)

    # ── Evaluate ──────────────────────────────────────────────────────
    print(f"\n[cv_real_eval] Evaluating {len(samples)} samples...")
    per_sample_results = []

    for i, sample in enumerate(samples):
        try:
            result = evaluate_sample(scanner, sample)
            per_sample_results.append(result)
        except Exception as exc:
            print(f"  [WARN] Sample {sample['image_id']} failed: {exc}")
            continue

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"  Progress: {i + 1}/{len(samples)} samples evaluated")

    # ── Aggregate ─────────────────────────────────────────────────────
    agg = aggregate_metrics(per_sample_results)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<25} {'Value':>12}")
    print(f"  {'-' * 40}")
    for key in ["macro_f1", "f1_title", "f1_variety", "f1_country",
                "year_exact_match", "price_match", "cer",
                "detection_rate", "avg_latency_ms", "p95_latency_ms"]:
        val = agg.get(key, 0)
        unit = "ms" if "latency" in key else ""
        print(f"  {key:<25} {val:>12.4f}{unit}")
    print(f"{'=' * 65}")

    # ── Save results ──────────────────────────────────────────────────
    save_json_results(agg, per_sample_results, args.output)
    generate_markdown_report(agg, per_sample_results, args.report)

    print(f"\n[cv_real_eval] Evaluation complete!")
    return agg, per_sample_results


if __name__ == "__main__":
    main()
