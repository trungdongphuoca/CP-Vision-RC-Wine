"""
gnn_retrieval.py — GNN Embedding Retrieval Module
===================================================
Loads pre-computed LightGCN wine embeddings produced by
``evaluation/gnn_indexer.py`` and exposes a lightweight cosine-similarity
retrieval function for hybrid (ChromaDB + GNN) recommendation.

Typical usage inside ``inference_rag.py``::

    from gnn_retrieval import GNNRetriever
    gnn = GNNRetriever()                       # loads embeddings + wine metadata
    results = gnn.retrieve(query_text, top_k=5) # returns list[dict]

If the embedding files are missing the module degrades gracefully —
all public methods return empty results and ``is_available`` is ``False``.
"""

import sys, os, logging
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_src_dir.parent))

import config as cfg

import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional

logger = logging.getLogger("gnn_retrieval")


class GNNRetriever:
    """Cosine-similarity retriever backed by pre-computed LightGCN embeddings.

    Parameters
    ----------
    embeddings_path : Path, optional
        ``.npy`` file with shape ``(N, D)`` wine embeddings.
    tfidf_path : Path, optional
        Pickled ``TfidfVectorizer`` (used to project query text into the
        same TF-IDF space the GNN was initialised with).
    svd_path : Path, optional
        Pickled ``TruncatedSVD`` (reduces TF-IDF to *D* dims).
    catalog_csv : Path, optional
        Wine catalog CSV for wine metadata lookup.
    """

    def __init__(
        self,
        embeddings_path: Optional[Path] = None,
        tfidf_path: Optional[Path] = None,
        svd_path: Optional[Path] = None,
        catalog_csv: Optional[Path] = None,
    ):
        self.is_available: bool = False
        self._embeddings: Optional[np.ndarray] = None      # (N, D)
        self._norms: Optional[np.ndarray] = None            # (N,)
        self._tfidf = None
        self._svd = None
        self._catalog_df: Optional[pd.DataFrame] = None
        self._index_map: Optional[np.ndarray] = None        # maps row → original df index

        # Resolve default paths via config
        embeddings_path = embeddings_path or cfg.RESULTS / "gnn_wine_embeddings.npy"
        tfidf_path = tfidf_path or cfg.RESULTS / "gnn_tfidf.pkl"
        svd_path = svd_path or cfg.RESULTS / "gnn_svd.pkl"
        catalog_csv = catalog_csv or cfg.WINE_CSV

        self._load(embeddings_path, tfidf_path, svd_path, catalog_csv)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(
        self,
        embeddings_path: Path,
        tfidf_path: Path,
        svd_path: Path,
        catalog_csv: Path,
    ) -> None:
        """Attempt to load all artefacts; set ``is_available`` accordingly."""
        try:
            if not embeddings_path.exists():
                logger.warning(
                    "[GNN] Embeddings not found at %s, using ChromaDB-only retrieval",
                    embeddings_path,
                )
                return

            # 1. Load embeddings
            self._embeddings = np.load(str(embeddings_path)).astype(np.float32)
            # Pre-compute L2 norms for fast cosine similarity
            self._norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            self._norms = np.where(self._norms == 0, 1e-9, self._norms)
            logger.info(
                "[GNN] Loaded embeddings: shape=%s", self._embeddings.shape
            )

            # 2. Load TF-IDF vectorizer + SVD (for query projection)
            if not tfidf_path.exists() or not svd_path.exists():
                logger.warning(
                    "[GNN] TF-IDF/SVD artefacts missing — query projection disabled"
                )
                return

            with open(str(tfidf_path), "rb") as f:
                self._tfidf = pickle.load(f)
            with open(str(svd_path), "rb") as f:
                self._svd = pickle.load(f)

            # 3. Load wine catalog (same filtering as gnn_indexer.py)
            if not catalog_csv.exists():
                logger.warning("[GNN] Catalog CSV not found at %s", catalog_csv)
                return

            df = pd.read_csv(str(catalog_csv))
            df = df.dropna(subset=["country", "variety", "description", "title"])
            df = df.reset_index(drop=True)
            self._catalog_df = df

            # Validate alignment
            if len(df) != self._embeddings.shape[0]:
                logger.warning(
                    "[GNN] Catalog rows (%d) != embedding rows (%d). "
                    "Re-run gnn_indexer.py to regenerate.",
                    len(df),
                    self._embeddings.shape[0],
                )
                self._embeddings = None
                return

            self.is_available = True
            logger.info(
                "[GNN] Retriever ready — %d wines, %d-dim embeddings",
                self._embeddings.shape[0],
                self._embeddings.shape[1],
            )

        except Exception as exc:
            logger.error("[GNN] Failed to initialise retriever: %s", exc)
            self.is_available = False

    def _query_to_embedding(self, query_text: str) -> np.ndarray:
        """Project a free-text query into the GNN embedding space.

        Pipeline mirrors gnn_indexer.py:
            text → TF-IDF → TruncatedSVD → 128-D vector
        """
        tfidf_vec = self._tfidf.transform([query_text])
        svd_vec = self._svd.transform(tfidf_vec)
        return svd_vec.astype(np.float32).reshape(1, -1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict]:
        """Return the *top_k* wines most similar to *query_text* by cosine
        similarity in GNN embedding space.

        Parameters
        ----------
        query_text : str
            Natural-language wine query (e.g. ``"bold Cabernet from Napa"``).
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict has keys ``title``, ``country``, ``variety``,
            ``price``, ``description``, ``gnn_score`` (float 0-1).
            Empty list if GNN retrieval is unavailable.
        """
        if not self.is_available:
            return []

        try:
            q_emb = self._query_to_embedding(query_text)       # (1, D)
            q_norm = np.linalg.norm(q_emb)
            if q_norm == 0:
                return []
            q_emb_normed = q_emb / q_norm                      # unit vector

            # Cosine similarity against all wine embeddings
            emb_normed = self._embeddings / self._norms         # (N, D)
            cos_sim = (emb_normed @ q_emb_normed.T).squeeze()   # (N,)

            # Top-K indices (descending similarity)
            top_indices = np.argsort(cos_sim)[::-1][:top_k]

            results: List[Dict] = []
            for idx in top_indices:
                row = self._catalog_df.iloc[idx]
                results.append({
                    "title": str(row.get("title", "")),
                    "country": str(row.get("country", "")),
                    "variety": str(row.get("variety", "")),
                    "price": float(row["price"]) if pd.notna(row.get("price")) else None,
                    "description": str(row.get("description", "")),
                    "gnn_score": float(cos_sim[idx]),
                })
            return results

        except Exception as exc:
            logger.error("[GNN] Retrieval failed: %s", exc)
            return []

    def get_embedding(self, wine_index: int) -> Optional[np.ndarray]:
        """Return the raw GNN embedding for a wine by its catalog index."""
        if not self.is_available or self._embeddings is None:
            return None
        if 0 <= wine_index < self._embeddings.shape[0]:
            return self._embeddings[wine_index]
        return None


# ── Module-level convenience ──────────────────────────────────────────────────

_default_retriever: Optional[GNNRetriever] = None


def get_retriever() -> GNNRetriever:
    """Lazily instantiate and return the module-level ``GNNRetriever``."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = GNNRetriever()
    return _default_retriever


def gnn_retrieve(query_text: str, top_k: int = 5) -> List[Dict]:
    """Convenience wrapper — drop-in function for hybrid retrieval.

    Returns an empty list when GNN embeddings are not available.
    """
    return get_retriever().retrieve(query_text, top_k=top_k)
