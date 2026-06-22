"""
evaluation/eval_logger.py
=========================
Module tu dong luu evaluation log theo chuan JSON.
Ap dung cho:
  - Bai toan Goi y Ruou (Wine Recommendation): CW-EASE^R+IPS
  - Bai toan Trich xuat Thuc the nhan chai (OCR Entity Extraction): Florence-2 + YOLO

Chuan log dam bao 4 tieu chi:
  1. Dinh danh  : run_id, timestamp, model name & version
  2. Tai hien   : git_commit, dataset_version, random_seed
  3. Ket qua    : metrics dac thu tung task
  4. Hieu nang  : inference_latency_ms, throughput
"""

import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Helpers ------------------------------------------------------------------

def _get_git_commit() -> str:
    """Lay hash git commit hien tai (7 ky tu). Tra ve 'N/A' neu khong co git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def _get_git_branch() -> str:
    """Lay ten git branch hien tai."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def _now_iso() -> str:
    """Timestamp ISO-8601 theo UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Core Logger Class --------------------------------------------------------

class EvalLogger:
    """
    Evaluation Logger -- luu log danh gia mo hinh ML duoi dang JSON co cau truc.

    Vi du su dung co ban:
    ----------------------
        logger = EvalLogger(
            model_name="CW-EASE^R",
            model_version="1.0.0",
            task="wine_recommendation",
            dataset_version="XWines-150K-v1",
            random_seed=42,
            output_dir="results/logs"
        )
        logger.start_run()
        # ... chay model ...
        logger.log_recommendation_metrics(recall={10: 0.257}, ndcg={10: 0.242}, f1={10: 0.086})
        logger.log_latency(total_ms=34800, per_sample_ms=3.78, n_samples=9200)
        logger.finish_run()
        path = logger.save()
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        task: str,
        dataset_version: str,
        random_seed: int,
        output_dir: str = "results/logs",
        run_id: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self.run_id          = run_id or str(uuid.uuid4())[:8]
        self.model_name      = model_name
        self.model_version   = model_version
        self.task            = task
        self.dataset_version = dataset_version
        self.random_seed     = random_seed
        self.output_dir      = Path(output_dir)
        self.extra_params    = extra_params or {}

        self._start_time: Optional[float] = None
        self._end_time:   Optional[float] = None
        self._metrics:    Dict[str, Any]  = {}
        self._latency:    Dict[str, Any]  = {}
        self._notes:      List[str]       = []

        # Python logger (stdout + file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / f"{self.run_id}_run.log"
        self._logger = logging.getLogger(f"EvalLogger.{self.run_id}")
        self._logger.setLevel(logging.DEBUG)
        if not self._logger.handlers:
            fmt = logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            self._logger.addHandler(ch)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)

        self._logger.info(
            f"EvalLogger initialized | run_id={self.run_id} | "
            f"model={model_name} v{model_version} | task={task}"
        )

    # -- Lifecycle -------------------------------------------------------------

    def start_run(self) -> "EvalLogger":
        """Danh dau bat dau luot chay evaluation."""
        self._start_time = time.perf_counter()
        self._logger.info(f"{'='*60}")
        self._logger.info(f"RUN STARTED  run_id={self.run_id}  task={self.task}")
        self._logger.info(f"{'='*60}")
        return self

    def finish_run(self) -> "EvalLogger":
        """Danh dau ket thuc luot chay evaluation."""
        self._end_time = time.perf_counter()
        elapsed = self._end_time - (self._start_time or self._end_time)
        self._logger.info(f"RUN FINISHED | total_elapsed={elapsed:.3f}s")
        return self

    def add_note(self, note: str) -> "EvalLogger":
        """Them ghi chu tu do vao log."""
        self._notes.append(note)
        self._logger.info(f"[NOTE] {note}")
        return self

    # -- Metrics: Wine Recommendation ------------------------------------------

    def log_recommendation_metrics(
        self,
        recall:    Dict[int, float],
        ndcg:      Dict[int, float],
        precision: Dict[int, float],
        f1:        Dict[int, float],
        n_users:   int,
        threshold: float = 4.0,
        baseline_recall_at10: Optional[float] = None,
    ) -> "EvalLogger":
        """
        Log metrics cho bai toan Goi y Ruou.
        Metrics: Recall@K, nDCG@K, Precision@K, F1@K
        """
        k_values = sorted(set(list(recall) + list(ndcg)))
        rows = {}
        for k in k_values:
            rows[f"@{k}"] = {
                "Precision": round(precision.get(k, 0.0), 6),
                "Recall":    round(recall.get(k, 0.0),    6),
                "F1":        round(f1.get(k, 0.0),        6),
                "nDCG":      round(ndcg.get(k, 0.0),      6),
            }

        self._metrics["recommendation"] = {
            "task_description": "Top-K wine recommendation from user rating history",
            "eval_protocol":    "RatioSplit 80/20, seed=42",
            "relevance_threshold": threshold,
            "n_eval_users":    n_users,
            "metrics_by_k":    rows,
        }

        if baseline_recall_at10 is not None:
            delta = recall.get(10, 0.0) - baseline_recall_at10
            self._metrics["recommendation"]["delta_recall_at10_vs_baseline"] = round(delta, 6)
            self._logger.info(
                f"[Recommendation] R@10={recall.get(10,0)*100:.4f}%  "
                f"nDCG@10={ndcg.get(10,0)*100:.4f}%  "
                f"delta_R@10={delta*100:+.4f}pp vs baseline"
            )
        else:
            self._logger.info(
                f"[Recommendation] R@10={recall.get(10,0)*100:.4f}%  "
                f"nDCG@10={ndcg.get(10,0)*100:.4f}%"
            )
        return self

    # -- Metrics: OCR Entity Extraction ----------------------------------------

    def log_ocr_metrics(
        self,
        entity_results: Dict[str, Dict[str, float]],
        n_samples: int,
        config: str = "unwarped",
        ocr_model: str = "Florence-2",
        detection_model: str = "YOLO11-wine",
    ) -> "EvalLogger":
        """
        Log metrics cho bai toan Trich xuat Thuc the nhan chai.
        Metrics: Precision / Recall / F1 theo tung entity.
        """
        macro_f1 = sum(v["f1"] for v in entity_results.values()) / max(len(entity_results), 1)

        self._metrics["ocr_entity_extraction"] = {
            "task_description":  "Named entity extraction from wine bottle label images",
            "ocr_model":         ocr_model,
            "detection_model":   detection_model,
            "image_config":      config,
            "n_samples":         n_samples,
            "entity_metrics": {
                k: {m: round(v, 4) for m, v in scores.items()}
                for k, scores in entity_results.items()
            },
            "macro_avg_f1": round(macro_f1, 4),
        }

        for entity, scores in entity_results.items():
            self._logger.info(
                f"[OCR/{config}] {entity:<10}  "
                f"P={scores['precision']:.4f}  R={scores['recall']:.4f}  F1={scores['f1']:.4f}"
            )
        self._logger.info(f"[OCR/{config}] Macro-avg F1 = {macro_f1:.4f}  (n_samples={n_samples})")
        return self

    # -- System Performance ----------------------------------------------------

    def log_latency(
        self,
        total_ms: float,
        per_sample_ms: float,
        n_samples: int,
        stage_breakdown: Optional[Dict[str, float]] = None,
    ) -> "EvalLogger":
        """Log hieu nang he thong: thoi gian xu ly, throughput."""
        throughput = (n_samples / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        self._latency = {
            "total_ms":         round(total_ms, 2),
            "per_sample_ms":    round(per_sample_ms, 4),
            "throughput_per_s": round(throughput, 2),
            "n_samples":        n_samples,
        }
        if stage_breakdown:
            self._latency["stage_breakdown_ms"] = {k: round(v, 2) for k, v in stage_breakdown.items()}

        self._logger.info(
            f"[Latency] total={total_ms:.0f}ms  "
            f"per_sample={per_sample_ms:.2f}ms  "
            f"throughput={throughput:.1f} samples/s"
        )
        return self

    # -- Build & Save ----------------------------------------------------------

    def _build_log(self) -> Dict[str, Any]:
        """Tong hop toan bo thong tin thanh dict JSON-ready."""
        elapsed = None
        if self._start_time and self._end_time:
            elapsed = round(self._end_time - self._start_time, 3)

        return {
            # 1. Dinh danh
            "identity": {
                "run_id":        self.run_id,
                "timestamp_utc": _now_iso(),
                "model": {
                    "name":    self.model_name,
                    "version": self.model_version,
                    "task":    self.task,
                },
            },
            # 2. Tai hien
            "reproducibility": {
                "git_commit":      _get_git_commit(),
                "git_branch":      _get_git_branch(),
                "dataset_version": self.dataset_version,
                "random_seed":     self.random_seed,
                "python_version":  platform.python_version(),
                "platform":        platform.system() + " " + platform.release(),
                "extra_params":    self.extra_params,
            },
            # 3. Ket qua
            "results": self._metrics,
            # 4. Hieu nang he thong
            "system_performance": {
                "wall_clock_seconds": elapsed,
                "inference_latency":  self._latency,
            },
            # Meta
            "notes":              self._notes,
            "log_schema_version": "1.0",
        }

    def save(self, filename: Optional[str] = None) -> Path:
        """Ghi log ra file JSON. Tra ve duong dan file da luu."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.model_name.replace(" ", "_").replace("^", "").replace("+", "")
            filename = f"eval_{safe_name}_{ts}_{self.run_id}.json"
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._build_log(), f, ensure_ascii=False, indent=2)
        self._logger.info(f"[SAVED] Eval log => {out_path}")
        return out_path

    def get_dict(self) -> Dict[str, Any]:
        """Tra ve log duoi dang dict."""
        return self._build_log()


# --- Context Manager ----------------------------------------------------------

class eval_run:
    """
    Context manager tien loi -- tu dong start/finish/save.

    Vi du:
        with eval_run(logger) as run:
            run.log_recommendation_metrics(...)
            run.log_latency(...)
        # -> file JSON duoc luu tu dong khi thoat block
    """
    def __init__(self, logger: EvalLogger):
        self.logger = logger

    def __enter__(self) -> EvalLogger:
        self.logger.start_run()
        return self.logger

    def __exit__(self, *_):
        self.logger.finish_run()
        self.logger.save()


# --- Demo ---------------------------------------------------------------------

if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    OUT  = ROOT / "results" / "logs"

    print("=" * 68)
    print("  DEMO: EvalLogger -- Ghi log danh gia mo hinh theo chuan JSON")
    print("=" * 68)

    # Demo 1: CW-EASE^R (Wine Recommendation)
    print("\n[Demo 1] Wine Recommendation -- CW-EASE^R+IPS+Ensemble\n")
    rec_logger = EvalLogger(
        model_name      = "CW-EASE^R",
        model_version   = "1.0.0",
        task            = "wine_recommendation",
        dataset_version = "XWines-Slim-150K-v1",
        random_seed     = 42,
        output_dir      = str(OUT),
        extra_params    = {
            "confidence_scheme": "soft_binary",
            "lambda":            100,
            "ips_beta":          0.3,
            "ensemble_lambdas":  [100, 350],
        }
    )
    with eval_run(rec_logger) as run:
        time.sleep(0.05)
        run.log_recommendation_metrics(
            recall    = {5: 0.1665, 10: 0.2566, 50: 0.5526, 100: 0.6898},
            ndcg      = {5: 0.2002, 10: 0.2412, 50: 0.3027, 100: 0.3144},
            precision = {5: 0.0684, 10: 0.0536, 50: 0.0238, 100: 0.0149},
            f1        = {5: 0.0925, 10: 0.0857, 50: 0.0451, 100: 0.0291},
            n_users   = 9200,
            threshold = 4.0,
            baseline_recall_at10 = 0.2504,
        )
        run.log_latency(
            total_ms      = 34800,
            per_sample_ms = 3.78,
            n_samples     = 9200,
            stage_breakdown = {
                "data_load_ms":     90,
                "matrix_build_ms":  60,
                "grid_search_ms":   28500,
                "eval_ms":          550,
                "ips_ensemble_ms":  5600,
            }
        )
        run.add_note("Best scheme: soft_binary outperforms binary_pos at R@10 by +1.0pp")
        run.add_note("IPS beta=0.3 chosen via ablation on validation set")

    # Demo 2: OCR Entity Extraction
    print("\n[Demo 2] OCR Entity Extraction -- Florence-2 + YOLO11-wine\n")
    ocr_logger = EvalLogger(
        model_name      = "Florence-2-OCR",
        model_version   = "large-ft-1.0",
        task            = "ocr_entity_extraction",
        dataset_version = "WineLabel-Curated-15-v1",
        random_seed     = 0,
        output_dir      = str(OUT),
        extra_params    = {
            "vlm":              "microsoft/Florence-2-large",
            "detector":         "YOLO11-wine-finetuned",
            "unwarping":        True,
            "unwarping_method": "cylindrical_projection",
        }
    )
    with eval_run(ocr_logger) as run:
        time.sleep(0.02)
        run.log_ocr_metrics(
            entity_results = {
                "Variety": {"precision": 0.4000, "recall": 0.4000, "f1": 0.4000},
                "Country": {"precision": 0.6000, "recall": 0.6000, "f1": 0.6000},
                "Winery":  {"precision": 0.3333, "recall": 0.3333, "f1": 0.3333},
                "Vintage": {"precision": 0.8000, "recall": 0.8000, "f1": 0.8000},
                "Label":   {"precision": 0.7333, "recall": 0.7333, "f1": 0.7333},
            },
            n_samples = 15,
            config    = "unwarped",
        )
        run.log_latency(
            total_ms      = 2607730,
            per_sample_ms = 2590,
            n_samples     = 1007,
            stage_breakdown = {
                "yolo_detection_ms":     120,
                "cylindrical_unwarp_ms": 280,
                "florence_ocr_ms":       2100,
                "entity_parse_ms":       90,
            }
        )
        run.add_note("Unwarped config improves macro-F1 vs curved by ~8pp on curated set")
        run.add_note("Full eval on 1007-image XWines-1K test set; curated n=15 for entity analysis")

    # In mau log JSON
    print("\n" + "=" * 68)
    print("  FILE MAU CAU TRUC LOG JSON (CW-EASE^R run):")
    print("=" * 68)
    print(json.dumps(rec_logger.get_dict(), ensure_ascii=False, indent=2))
    print("\n" + "=" * 68)
    print(f"  Logs da luu vao: {OUT}")
    print("=" * 68)
