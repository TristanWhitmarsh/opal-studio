"""
Dimensionality reduction helpers for visualising clustering results.

Functions return 2-D NumPy arrays of shape (n_cells, 2).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def parallel_analysis_n_components(
    data: np.ndarray,
    n_permutations: int = 100,
    percentile: float = 95.0,
    random_state: int = 42,
) -> int:
    """Horn's parallel analysis — Method 2 (shuffle correlation matrix).

    Compute the correlation matrix of the data once, then for each
    permutation shuffle its off-diagonal elements (keeping the diagonal
    at 1) to create a random correlation matrix with the same marginal
    correlation magnitudes but no structure.  Eigenvalues of these random
    matrices form the reference distribution.

    This is more elegant and much faster than the raw-data permutation
    approach because only an (n_channels × n_channels) matrix is touched
    per permutation rather than the full (n_cells × n_channels) dataset.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    n_permutations : int
        Number of shuffles (default 100).
    percentile : float
        Threshold percentile of the random eigenvalue distribution
        (default 95 — conservative; use 50 for the original Horn criterion).
    random_state : int

    Returns
    -------
    int
        Number of components to retain (≥ 1).
    """
    from sklearn.preprocessing import StandardScaler

    # Standardise so PA operates on the correlation matrix regardless of
    # which normalisation was applied upstream.
    data_sc = StandardScaler().fit_transform(data)
    n_features = data_sc.shape[1]

    # Real correlation matrix and its eigenvalues (descending)
    corr = np.corrcoef(data_sc, rowvar=False)
    real_ev = np.linalg.eigvalsh(corr)[::-1]

    # Off-diagonal indices (lower triangle)
    rows, cols = np.tril_indices(n_features, k=-1)
    off_diag = corr[rows, cols].copy()

    rng = np.random.default_rng(random_state)
    rand_ev = np.empty((n_permutations, n_features), dtype=np.float64)
    for i in range(n_permutations):
        shuffled = off_diag.copy()
        rng.shuffle(shuffled)
        rand_corr = np.eye(n_features)
        rand_corr[rows, cols] = shuffled
        rand_corr[cols, rows] = shuffled          # enforce symmetry
        rand_ev[i] = np.linalg.eigvalsh(rand_corr)[::-1]

    thresholds = np.percentile(rand_ev, percentile, axis=0)
    n_comp = int(np.sum(real_ev > thresholds))
    n_comp = max(1, min(n_comp, n_features))

    logger.info(
        "Parallel analysis: %d/%d components retained "
        "(%d shuffles, %.0fth percentile)",
        n_comp, n_features, n_permutations, percentile,
    )
    return n_comp


def run_tsne(
    data: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Run t-SNE on *data* (n_cells × n_features) and return 2-D embedding."""
    from sklearn.manifold import TSNE

    n_samples = data.shape[0]
    actual_perplexity = min(perplexity, max(1, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(data)


def run_umap(
    data: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray | None:
    """Run UMAP on *data* and return 2-D embedding, or None if umap-learn is absent."""
    try:
        import umap as umap_lib
    except ImportError:
        return None

    n_samples = data.shape[0]
    actual_neighbors = min(n_neighbors, max(2, n_samples - 1))

    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(data)
