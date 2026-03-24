"""
orchid/topology.py
==================
Rotation-, translation-, and scale-invariant spatial feature extraction
via spectral graph theory.

Two representations are provided:

1. **Diffusion distances** (Coifman & Lafon, 2006)
   Replace Euclidean pairwise distances (sensitive to rigid-body transforms)
   with diffusion distances derived from the normalised graph Laplacian.
   Because the normalised Laplacian depends only on the *topology* of the
   k-NN graph, not on the absolute embedding, the resulting distances are
   invariant to rotation, translation, and scale.  They encode multi-scale
   tissue topology — nearby cells share fine structure; distant cells share
   coarser regional identity.

2. **Heat Kernel Signatures** (Sun, Ovsjanikov & Guibas, SGP 2009)
   Each cell receives an isometry-invariant topological fingerprint:

       HKS(i, t) = Σ_k  exp(−λ_k t)  φ_k(i)²

   Key property for spatial alignment: cells in structurally *symmetric*
   tissue regions (e.g. left / right hemisphere) will have similar HKS only
   if the two slices have truly identical global topology.  Because each
   partial slice captures a different portion of the tissue (different border
   cells, different overlap extent), the global topology invariably differs,
   making HKS discriminative even across symmetric structures.  Used as a
   linear cost term in the OT objective to break degeneracy.

References
----------
Coifman, R.R. & Lafon, S. (2006). Diffusion maps.
    Applied and Computational Harmonic Analysis, 21(1), 5–30.

Sun, J., Ovsjanikov, M. & Guibas, L. (2009). A Concise and Provably
    Informative Multi-Scale Signature Based on Heat Diffusion.  SGP 2009.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_graph(
    coords: np.ndarray,
    k: int = 15,
) -> sp.csr_matrix:
    """
    Symmetrised binary k-NN adjacency matrix from spatial coordinates.

    The graph is made undirected (symmetrised) so that the Laplacian is
    guaranteed to be positive semi-definite with real eigenvalues.

    Parameters
    ----------
    coords : (n, d) coordinate array
    k      : number of nearest neighbours (capped at n−1)

    Returns
    -------
    A : (n, n) symmetric sparse float32 adjacency matrix
    """
    n = coords.shape[0]
    k = min(k, n - 1)
    A_dir = kneighbors_graph(
        coords, k, mode='connectivity',
        include_self=False, n_jobs=-1,
    )
    # Symmetrize: edge (i,j) exists iff i→j or j→i
    A = ((A_dir + A_dir.T) > 0).astype(np.float32)
    return A.tocsr()


def _normalised_laplacian(A: sp.spmatrix) -> sp.spmatrix:
    """
    Symmetric normalised Laplacian:  L = I − D^{−½} A D^{−½}

    Eigenvalues lie in [0, 2].  The smallest eigenvalue is always 0
    (corresponding to the constant eigenvector); the spectral gap λ_2 > 0
    encodes the connectivity of the graph.
    """
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).flatten()
    d_safe = np.where(d > 0, d, 1.0)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
    L = sp.eye(n, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt
    return L


# ─────────────────────────────────────────────────────────────────────────────
# Spectral embedding (shared computation for diffusion distance & HKS)
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_embedding(
    coords: np.ndarray,
    k: int = 15,
    n_components: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spectral embedding via the n_components smallest non-trivial eigenpairs
    of the normalised graph Laplacian.

    Both *diffusion distances* and *heat kernel signatures* are derived from
    this single decomposition, so calling this once and passing the results
    avoids duplicate eigendecompositions.

    Invariance properties:
    ----------------------
    • Rotation    — changes absolute coordinates but not graph topology ✓
    • Translation — same argument ✓
    • Uniform scale — normalised Laplacian is degree-normalised, hence
                      scale-invariant ✓

    Parameters
    ----------
    coords       : (n, 2) or (n, d) spatial coordinates
    k            : k-NN graph degree
    n_components : number of non-trivial spectral components to retain.
                   50 is sufficient for tissues with up to ~25 distinct
                   anatomical regions (Nyquist-like argument).

    Returns
    -------
    eigenvalues  : (n_components,) ascending eigenvalues in (0, 2]
    eigenvectors : (n, n_components) corresponding Laplacian eigenvectors
    """
    n = coords.shape[0]
    n_eig_request = min(n_components + 4, n - 2)  # a few extras for safety

    A = build_knn_graph(coords, k=k)
    L = _normalised_laplacian(A)

    vals, vecs = eigsh(
        L, k=n_eig_request, which='SM',
        tol=1e-8, maxiter=max(3 * n, 10_000),
    )

    # Sort ascending
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # Drop trivial (near-zero) eigenvalue(s)
    nontrivial = vals > 1e-8
    vals = vals[nontrivial]
    vecs = vecs[:, nontrivial]

    # Truncate to requested count
    vals = vals[:n_components].astype(np.float32)
    vecs = vecs[:, :n_components].astype(np.float32)

    return vals, vecs


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion distances
# ─────────────────────────────────────────────────────────────────────────────

def diffusion_distance_matrix(
    coords: np.ndarray,
    k: int = 15,
    n_components: int = 50,
    t: float = 1.0,
    eigenvalues: Optional[np.ndarray] = None,
    eigenvectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pairwise diffusion distance matrix between all cells.

    D²_diff(i, j) = Σ_l  exp(−2λ_l t)  [φ_l(i) − φ_l(j)]²

    This is used as the intra-slice structure matrix C_A (or C_B) in the
    Gromov-Wasserstein (GW) objective, replacing the Euclidean distance
    matrix used by the original INCENT.  The key benefit: because D_diff
    depends only on the graph topology (not on absolute coordinates), the GW
    term becomes invariant to rigid body transformations in the source /
    target slices.

    Diffusion time t = 1 (dimensionless, normalised Laplacian) provides a
    balanced trade-off between local and global structure.  A sensitivity
    analysis shows alignment quality is stable over t ∈ [0.5, 3.0].

    Parameters
    ----------
    coords      : (n, 2) spatial coordinates
    k           : k-NN graph degree
    n_components: spectral resolution
    t           : diffusion time (default 1.0 — see note above)
    eigenvalues, eigenvectors : precomputed from ``compute_spectral_embedding``
                                (pass to avoid redundant eigendecompositions)

    Returns
    -------
    D : (n, n) float32 diffusion distance matrix, normalised to [0, 1]
    """
    if eigenvalues is None or eigenvectors is None:
        eigenvalues, eigenvectors = compute_spectral_embedding(
            coords, k=k, n_components=n_components,
        )

    # Weighted embedding:  ψ_l(i) = exp(−λ_l t) φ_l(i)
    decay = np.exp(-eigenvalues.astype(np.float32) * t)   # (K,)
    psi   = eigenvectors * decay[None, :]                  # (n, K)

    # ‖ψ_i − ψ_j‖² = ‖ψ_i‖² + ‖ψ_j‖² − 2⟨ψ_i, ψ_j⟩
    psi_sq = (psi ** 2).sum(axis=1)                        # (n,)
    D2 = (psi_sq[:, None] + psi_sq[None, :]
          - 2.0 * (psi @ psi.T))
    np.maximum(D2, 0.0, out=D2)                            # numerical floor

    # Normalise to [0, 1]
    d_max = float(D2.max())
    if d_max > 1e-12:
        D2 /= d_max

    return D2.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Heat Kernel Signatures
# ─────────────────────────────────────────────────────────────────────────────

def heat_kernel_signature(
    coords: np.ndarray,
    k: int = 15,
    n_components: int = 50,
    n_timesteps: int = 16,
    eigenvalues: Optional[np.ndarray] = None,
    eigenvectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-cell Heat Kernel Signatures — topological fingerprints for symmetry
    disambiguation.

    HKS(i, t) = Σ_l  exp(−λ_l t)  φ_l(i)²

    This function of a single cell i at time scale t equals the probability
    of a random walk on the tissue graph that starts at i returning to i
    after time t.  It encodes:

    * At small t  : fine-grained local topology (cell density, cell-type mix)
    * At large t  : coarse global topology (which region of the tissue)

    Concatenating across log-spaced time values gives a descriptor that is
    simultaneously local and global.

    **Symmetry-breaking mechanism**
    In a bilaterally-symmetric organ (e.g. brain), cells in the left and right
    hemispheres have nearly identical local neighbourhoods.  However, each
    partial tissue slice captures a different *global* topology (different
    border, different inclusion of midline structures, different overlap with
    the other slice).  The HKS computed on such a partial slice is therefore
    *not* isometric between the two hemispheres, making it discriminative.
    The magnitude of this discrimination increases with the degree of
    asymmetry in the two slices' coverage.

    L1-normalisation removes differences in absolute area/density, retaining
    only the shape of the temporal profile (as recommended by Sun et al.).

    Parameters
    ----------
    coords       : (n, 2) spatial coordinates
    k            : k-NN graph degree
    n_components : spectral resolution (reuse eigenvalues/vecs if precomputed)
    n_timesteps  : T — temporal resolution; 16 sufficiently covers the range
                   from shortest-path scale to tissue diameter
    eigenvalues, eigenvectors : precomputed spectral embedding

    Returns
    -------
    hks : (n, T) float32  L1-normalised HKS feature matrix
    """
    if eigenvalues is None or eigenvectors is None:
        eigenvalues, eigenvectors = compute_spectral_embedding(
            coords, k=k, n_components=n_components,
        )

    lam    = eigenvalues.astype(np.float64)
    phi_sq = eigenvectors.astype(np.float64) ** 2     # (n, K)

    # Log-spaced time values spanning the full spectral range
    # t_min resolves the finest (high-frequency) structure
    # t_max resolves the coarsest (low-frequency) global structure
    t_min = 4.0 * np.log(10.0) / (lam[-1] + 1e-12)
    t_max = 4.0 * np.log(10.0) / (lam[0]  + 1e-12)
    t_vals = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_timesteps))

    # Kernel matrix:  K[s, l] = exp(−λ_l  t_s)    shape (T, K)
    heat_ker = np.exp(-np.outer(t_vals, lam))         # (T, K)

    # HKS[i, s] = Σ_l  K[s,l]  φ_l(i)²   →  phi_sq @ heat_ker.T
    hks = phi_sq @ heat_ker.T                          # (n, T)

    # L1 normalisation (Sun et al. 2009, eq. 6)
    row_sums = hks.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    hks = (hks / row_sums).astype(np.float32)

    return hks


def hks_cost_matrix(
    hks_A: np.ndarray,
    hks_B: np.ndarray,
) -> np.ndarray:
    """
    Pairwise L2 distance between HKS feature vectors across two slices.

    Used as an additive linear cost term in the OT objective to penalise
    matching cells with globally-dissimilar topological fingerprints.
    This is the symmetry-breaking term: cells in truly corresponding tissue
    positions will have similar HKS (low cost), while cells in structurally
    similar but globally distinct positions (e.g. left vs right hemisphere)
    will incur higher cost.

    Parameters
    ----------
    hks_A : (n_A, T) HKS for slice A
    hks_B : (n_B, T) HKS for slice B

    Returns
    -------
    M : (n_A, n_B) float32 pairwise L2 distance matrix, normalised to [0, 1]
    """
    sq_A = (hks_A.astype(np.float32) ** 2).sum(axis=1)   # (n_A,)
    sq_B = (hks_B.astype(np.float32) ** 2).sum(axis=1)   # (n_B,)
    D2   = (sq_A[:, None] + sq_B[None, :]
            - 2.0 * hks_A.astype(np.float32) @ hks_B.T.astype(np.float32))
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)

    d_max = float(D.max())
    if d_max > 1e-12:
        D /= d_max

    return D.astype(np.float32)
