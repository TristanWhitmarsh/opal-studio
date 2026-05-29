"""
Cell clustering based on per-cell mean channel intensities.

Provides Leiden, Louvain, DBSCAN, KMeans, FlowSOM, PhenoGraph, and
Hierarchical clustering methods.
Input is a 2D array of shape (n_cells, n_channels) with per-cell mean
intensities. Returns an integer array of cluster labels (length n_cells).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _zscore(col):
    """Z-score a single column, handling constant columns."""
    std = col.std()
    if std == 0:
        return np.zeros_like(col)
    return (col - col.mean()) / std


def normalize_data(data, method="zscore", cofactor=5, skewness_threshold=1):
    """Normalize a (n_cells, n_channels) array.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    method : str
        "none"         – return data unchanged
        "zscore"       – per-channel zero-mean, unit-variance
        "minmax"       – per-channel scale to [0, 1]
        "log-z"        – log1p + Z-score for skewed channels, Z-score only
                         for channels with abs(skewness) <= *skewness_threshold*
        "yeo-johnson"  – Yeo-Johnson power transform + Z-score per channel
        "arcsinh"      – arcsinh(x / cofactor) + Z-score per channel
    cofactor : float
        Divisor used by the arcsinh transform (default 5).
    skewness_threshold : float
        Skewness cutoff for the log-z method (default 1).

    Returns
    -------
    ndarray, same shape as *data*
    """
    if method == "none":
        return data.copy()

    if method == "zscore":
        out = data.copy()
        for j in range(out.shape[1]):
            out[:, j] = _zscore(out[:, j])
        return out

    if method == "minmax":
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        rng = dmax - dmin
        rng[rng == 0] = 1.0
        return (data - dmin) / rng

    if method == "log-z":
        from scipy.stats import skew

        out = data.copy()
        for j in range(out.shape[1]):
            col = out[:, j]
            s = skew(col)
            if abs(s) > skewness_threshold:
                col = np.log1p(col)
                logger.info("Channel %d: log1p + Z-score (skewness %.2f)", j, s)
            else:
                logger.info("Channel %d: Z-score only (skewness %.2f)", j, s)
            out[:, j] = _zscore(col)
        return out

    if method == "yeo-johnson":
        from scipy.stats import yeojohnson

        out = data.copy()
        for j in range(out.shape[1]):
            col = out[:, j]
            mask = ~np.isnan(col)
            clean = col[mask]
            if len(np.unique(clean)) < 2:
                logger.info("Channel %d: constant — skipping Yeo-Johnson", j)
                out[:, j] = _zscore(col)
                continue
            try:
                transformed, lam = yeojohnson(clean)
                transformed = _zscore(transformed)
                result = np.full_like(col, np.nan, dtype=np.float64)
                result[mask] = transformed
                out[:, j] = result
                logger.info("Channel %d: Yeo-Johnson (lambda=%.2f)", j, lam)
            except Exception as e:
                logger.warning("Channel %d: Yeo-Johnson failed (%s), using Z-score", j, e)
                out[:, j] = _zscore(col)
        return out

    if method == "arcsinh":
        out = data.copy()
        for j in range(out.shape[1]):
            col = np.arcsinh(out[:, j] / cofactor)
            out[:, j] = _zscore(col)
            logger.info("Channel %d: arcsinh (cofactor=%g) + Z-score", j, cofactor)
        return out

    raise ValueError(f"Unknown normalization method: {method}")


def run_leiden(data, resolution=0.5):
    """Leiden community detection on a k-NN graph.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    resolution : float
        Higher values yield more (smaller) clusters.

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
    n_clusters : int
    """
    import scanpy as sc

    adata = sc.AnnData(data.astype(np.float32))
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution)
    labels = adata.obs["leiden"].astype(int).values
    n_clusters = len(np.unique(labels))
    logger.info("Leiden: %d clusters (resolution=%.2f)", n_clusters, resolution)
    return labels, n_clusters


def run_dbscan(data, eps=None, min_samples=10):
    """DBSCAN density-based clustering.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
        Should be PCA-reduced before calling; pass raw normalised data
        only if n_channels is small.
    eps : float or None
        Neighbourhood radius.  When None, estimated as the 90th percentile
        of each cell's distance to its min_samples-th nearest neighbour.
    min_samples : int
        Minimum points in a neighbourhood to form a core point.

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
        Noise points are labelled -1.
    n_clusters : int
        Number of clusters (excluding noise).
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    if eps is None or eps <= 0:
        k = min(min_samples, len(data) - 1)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(data)
        distances, _ = nn.kneighbors(data)
        eps = float(np.percentile(distances[:, -1], 90))
        logger.info("DBSCAN auto-eps: %.4f (90th pct of %d-NN distances)", eps, k)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    n_clusters = len(set(labels) - {-1})
    logger.info("DBSCAN: %d clusters (eps=%.4f, min_samples=%d)",
                n_clusters, eps, min_samples)
    return labels, n_clusters


def run_kmeans(data, n_clusters=5):
    """K-Means clustering.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    n_clusters : int

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
    n_clusters : int
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(data)
    logger.info("KMeans: %d clusters", n_clusters)
    return labels, n_clusters


def run_louvain(data, resolution=0.5):
    """Louvain community detection on a k-NN graph.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    resolution : float
        Higher values yield more (smaller) clusters.

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
    n_clusters : int
    """
    import scanpy as sc

    adata = sc.AnnData(data.astype(np.float32))
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, resolution=resolution)
    labels = adata.obs["louvain"].astype(int).values
    n_clusters = len(np.unique(labels))
    logger.info("Louvain: %d clusters (resolution=%.2f)", n_clusters, resolution)
    return labels, n_clusters


def run_phenograph(data, k=30):
    """PhenoGraph clustering (Levine et al., 2015).

    Builds a k-nearest-neighbour graph and applies Louvain community
    detection to discover phenotypically distinct populations.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    k : int
        Number of nearest neighbours for the k-NN graph (default 30).

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
    n_clusters : int
    """
    import phenograph

    communities, graph, Q = phenograph.cluster(data, k=k)
    labels = np.asarray(communities, dtype=int)
    n_clusters = len(set(labels) - {-1})
    logger.info("PhenoGraph: %d clusters (k=%d, modularity=%.4f)",
                n_clusters, k, Q)
    return labels, n_clusters


def run_flowsom(data, xdim=10, ydim=10, n_clusters=10):
    """FlowSOM-style clustering (Van Gassen et al., 2015).

    Step 1 – SOM quantisation: MiniBatchKMeans with (xdim * ydim) nodes
             assigns every cell to its nearest prototype.
    Step 2 – Metaclustering: AgglomerativeClustering groups the SOM node
             centres into *n_clusters* metaclusters.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    xdim, ydim : int
        SOM grid dimensions (default 10 x 10 = 100 nodes).
    n_clusters : int
        Number of metaclusters to produce (default 10).

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
    n_clusters : int
    """
    from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

    n_nodes = min(xdim * ydim, len(data))
    n_meta  = min(n_clusters, n_nodes)

    som = MiniBatchKMeans(n_clusters=n_nodes, random_state=42, n_init=3)
    cell_to_node = som.fit_predict(data)

    meta = AgglomerativeClustering(n_clusters=n_meta)
    node_labels = meta.fit_predict(som.cluster_centers_)

    labels = node_labels[cell_to_node]
    n_out  = len(np.unique(labels))
    logger.info("FlowSOM: %d metaclusters (grid=%dx%d)", n_out, xdim, ydim)
    return labels, n_out


def run_hierarchical(data, n_clusters=5, linkage="ward", metric="euclidean"):
    """Agglomerative hierarchical clustering.

    Parameters
    ----------
    data : ndarray, shape (n_cells, n_channels)
    n_clusters : int
        Number of clusters to produce.
    linkage : str
        Linkage criterion: 'ward', 'complete', 'average', or 'single'.
        Note: 'ward' only works with metric='euclidean'.
    metric : str
        Distance metric: 'euclidean', 'cosine', 'manhattan', etc.

    Returns
    -------
    labels : ndarray of int, shape (n_cells,)
    n_clusters : int
    """
    from sklearn.cluster import AgglomerativeClustering

    # Ward linkage requires euclidean metric
    if linkage == "ward" and metric != "euclidean":
        logger.warning("Ward linkage requires euclidean metric; overriding metric to 'euclidean'.")
        metric = "euclidean"

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric if linkage != "ward" else "euclidean",
    )
    labels = model.fit_predict(data)
    n_out = len(np.unique(labels))
    logger.info("Hierarchical: %d clusters (linkage=%s, metric=%s)",
                n_out, linkage, metric)
    return labels, n_out
