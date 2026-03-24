"""
orchid/biology.py
=================
Multi-scale biological feature extraction and cross-timepoint normalization.

Two components:

1. **Multi-scale neighborhood distributions**
   The original INCENT computes the cell-type composition of each cell's
   neighbourhood at a single user-specified radius.  This creates two
   problems: (a) the result is sensitive to the choice of radius, which is
   a non-trivial hyperparameter; (b) a single scale captures only one level
   of tissue organisation (local intercellular, functional unit, or regional),
   missing the hierarchical nature of tissue architecture.

   Here we compute neighbourhood distributions at *n_radii* log-spaced radii
   spanning the data's own spatial scale — from the median nearest-neighbour
   distance (intercellular) to 1/4 of the tissue's largest extent (regional).
   Radii are chosen data-adaptively, eliminating a key hyperparameter.
   The resulting Jensen-Shannon divergences are averaged across scales,
   equally weighting each level of tissue organisation.

2. **Cross-timepoint gene expression normalization**
   Gene expression undergoes systematic shifts between developmental
   timepoints: cell-type mean expressions change while the relative
   within-cell-type variation (which encodes spatial identity) is more
   stable.  Intra-cell-type z-scoring removes these global temporal shifts
   while retaining spatially-informative variation, enabling meaningful
   cosine distance comparisons across timepoints.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from anndata import AnnData
from sklearn.neighbors import BallTree, NearestNeighbors
from tqdm import tqdm
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Automatic scale selection
# ─────────────────────────────────────────────────────────────────────────────

def auto_select_radii(
    coords: np.ndarray,
    n_radii: int = 4,
) -> np.ndarray:
    """
    Data-driven selection of neighbourhood radii for multi-scale analysis.

    Radii are log-spaced from the intercellular scale (2× median nearest-
    neighbour distance) to the mesoscale (1/4 of the tissue's largest
    dimension).  This range covers the organisational hierarchy from
    individual cell contacts to functional tissue units.

    Using log-spacing rather than linear spacing gives equal weighting to
    each decade of spatial scale, matching the self-similar nature of
    tissue architecture.

    Parameters
    ----------
    coords  : (n, 2) spatial coordinates (any consistent unit)
    n_radii : number of radii  (4 adequately covers local → mesoscale)

    Returns
    -------
    radii : (n_radii,) sorted radii in the same units as coords
    """
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    d_nn = float(np.median(dists[:, 1]))                     # intercellular scale

    extent = float(np.max(coords.max(axis=0) - coords.min(axis=0)))

    r_min = max(2.0 * d_nn, 1e-6)
    r_max = max(extent / 4.0, r_min * 20.0)                 # fallback for tiny slices

    return np.exp(
        np.linspace(np.log(r_min), np.log(r_max), n_radii)
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale neighbourhood distributions
# ─────────────────────────────────────────────────────────────────────────────

def neighborhood_distribution_multi_scale(
    adata: AnnData,
    radii: np.ndarray,
    cell_type_order: List[str],
    smoothing: float = 0.01,
) -> np.ndarray:
    """
    Multi-scale cell-type neighbourhood distributions.

    For each cell i and each radius r_l, returns the normalised fraction of
    each cell type within an Euclidean ball of radius r_l centred at i.

    A small Laplace smoothing constant is added before normalisation to
    prevent zero-probability cell types (which would cause −∞ in log terms
    during JSD computation).

    Parameters
    ----------
    adata           : AnnData with .obsm['spatial'] and .obs['cell_type_annot']
    radii           : (L,) radii at which to evaluate the distribution
    cell_type_order : canonical ordering of cell types (must be identical
                      for sliceA and sliceB so that K-dimensions align)
    smoothing       : Laplace smoothing constant

    Returns
    -------
    nd : (n_cells, K, L) float32 normalised distributions
         nd[i, :, l].sum() == 1  for all i, l
    """
    cell_types = np.array(adata.obs['cell_type_annot'].astype(str))
    ct2idx     = {c: i for i, c in enumerate(cell_type_order)}
    K          = len(cell_type_order)

    coords = adata.obsm['spatial']
    n      = adata.shape[0]
    L      = len(radii)
    tree   = BallTree(coords)

    nd = np.zeros((n, K, L), dtype=np.float32)

    for l, r in enumerate(radii):
        neighbor_lists = tree.query_radius(coords, r=float(r))
        for i in range(n):
            for idx in neighbor_lists[i]:
                ct = cell_types[idx]
                if ct in ct2idx:
                    nd[i, ct2idx[ct], l] += 1.0

    # Laplace smoothing + normalisation
    nd += smoothing
    nd /= nd.sum(axis=1, keepdims=True)

    return nd


# ─────────────────────────────────────────────────────────────────────────────
# Jensen-Shannon divergence (row-by-row, memory-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _jsd_one_vs_all(p: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Jensen-Shannon divergence (not distance) between a single distribution p
    and every row of Q.

    JSD(p ‖ q) = H((p+q)/2) − (H(p) + H(q))/2

    Uses the entropy form to avoid the square root that would be needed for
    the Jensen-Shannon *distance*; the square root is taken in
    ``multi_scale_jsd_cost`` after averaging across scales, preserving the
    metric property of the final cost matrix.

    Memory: O(n_Q × K) per call — safe for large n_Q.

    Parameters
    ----------
    p : (K,) distribution
    Q : (n_Q, K) distributions

    Returns
    -------
    jsd : (n_Q,) non-negative divergence values
    """
    def _h(X: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return -np.where(X > 0, X * np.log(X + 1e-300), 0.0).sum(axis=-1)

    M = 0.5 * (p[None, :] + Q)      # (n_Q, K)
    return np.maximum(_h(M) - 0.5 * (_h(p) + _h(Q)), 0.0)


def multi_scale_jsd_cost(
    nd_A: np.ndarray,
    nd_B: np.ndarray,
) -> np.ndarray:
    """
    Multi-scale Jensen-Shannon *distance* cost matrix.

    For each spatial scale l independently, compute the full (n_A × n_B)
    JSD matrix between neighbourhood distributions.  Average across scales,
    then take the square root to obtain the JSD distance (which satisfies
    the metric axioms).

    Equal weighting across scales reflects the assumption that each level of
    tissue organisation contributes equally to spatial identity.

    Parameters
    ----------
    nd_A : (n_A, K, L) multi-scale distributions for slice A
    nd_B : (n_B, K, L) multi-scale distributions for slice B

    Returns
    -------
    M : (n_A, n_B) float32 mean-JSD-distance cost matrix, values in [0, 1]
    """
    n_A, K, L = nd_A.shape
    n_B        = nd_B.shape[0]

    M_sum = np.zeros((n_A, n_B), dtype=np.float64)

    for l in tqdm(range(L), desc="Multi-scale JSD", leave=False):
        P = nd_A[:, :, l].astype(np.float64)   # (n_A, K)
        Q = nd_B[:, :, l].astype(np.float64)   # (n_B, K)
        for i in range(n_A):
            M_sum[i, :] += _jsd_one_vs_all(P[i], Q)

    # Average across scales → JSD distance (square root, bounded in [0, 1])
    M_avg = M_sum / L
    np.maximum(M_avg, 0.0, out=M_avg)
    M_dist = np.sqrt(M_avg / np.log(2.0))      # normalise by max possible JSD

    return M_dist.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-timepoint gene expression normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_cross_timepoint(
    sliceA: AnnData,
    sliceB: AnnData,
    use_rep: Optional[str] = None,
) -> Tuple[AnnData, AnnData]:
    """
    Intra-cell-type z-score normalisation for cross-timepoint alignment.

    **Biological rationale**
    Between developmental timepoints, gene expression undergoes two types
    of change:
    (a) *Temporal drift*: the cell-type mean expression shifts as cells
        mature, respond to stimuli, or change state.  This is a systematic
        change that is largely shared across all cells of the same type,
        independent of their spatial position.
    (b) *Spatial variation*: within a cell type, gene expression varies with
        spatial position (e.g. dorsal vs ventral, medial vs lateral gradient
        genes).  This variation is the signal we want to exploit for alignment
        and is thought to be more temporally stable than absolute expression.

    By z-scoring within each cell type independently for each slice, we
    remove component (a) while preserving component (b).  The resulting
    z-scores represent each cell's *relative* deviation from its cell-type
    mean — a more temporally robust representation for cross-timepoint
    comparisons.

    **Important**: z-scoring is applied *per slice* (not pooled), so it
    removes intra-slice temporal drift, not cross-slice differences.

    Parameters
    ----------
    sliceA, sliceB : AnnData with .obs['cell_type_annot']
    use_rep        : key in .obsm for expression matrix (None → use .X)

    Returns
    -------
    sliceA_z, sliceB_z : copies with z-scored expression
    """
    def _zscore(adata: AnnData) -> AnnData:
        adata = adata.copy()

        if use_rep is not None:
            X = adata.obsm[use_rep].copy().astype(np.float32)
        else:
            X = (adata.X.toarray()
                 if scipy.sparse.issparse(adata.X)
                 else np.array(adata.X)).astype(np.float32)

        labels = adata.obs['cell_type_annot'].values

        for ct in np.unique(labels):
            mask = labels == ct
            if mask.sum() < 2:
                continue
            sub  = X[mask]
            mean = sub.mean(axis=0)
            std  = sub.std(axis=0)
            std  = np.where(std < 1e-8, 1.0, std)     # stabilise genes with no variation
            X[mask] = (sub - mean) / std

        if use_rep is not None:
            adata.obsm[use_rep] = X
        else:
            adata.X = X

        return adata

    return _zscore(sliceA), _zscore(sliceB)
