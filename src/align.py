"""
orchid/align.py
===============
ORCHID: Orientation-Robust, Cross-resolution, Hierarchical Integration
        for spatial transcriptomics Data

This module provides the single public entry point:

    pi, R, t = pairwise_align_orchid(sliceA, sliceB, alpha)

The function orchestrates six components:

  I.   Preprocessing (shared genes / cell types; optional z-scoring)
  II.  Spectral embedding (shared computation for D and HKS)
  III. Diffusion distance matrices  — rotation/scale-invariant GW matrices
  IV.  Heat Kernel Signature cost   — symmetry-breaking linear cost
  V.   Gene + multi-scale JSD cost  — biological linear cost
  VI.  Adaptive FUGW solve          — unbalanced OT for partial overlap
  VII. Iterative rigid refinement   — explicit rotation/translation recovery

Design philosophy
-----------------
• Minimal exposed hyperparameters: only *alpha* is truly problem-specific.
  All others have well-motivated defaults that are robust across datasets.
• Caching at every stage: expensive matrices (spectral decompositions,
  JSD, HKS) are saved to filePath and reused on re-runs.
• Memory budget: all large matrices are float32; the n×n diffusion
  distance matrices are the dominant cost (~900 MB for n=15 k cells).
"""

from __future__ import annotations

import os
import time
import datetime
import numpy as np
import pandas as pd
import scipy.sparse
import ot
from anndata import AnnData
from typing import List, Optional, Tuple, Union
from numpy.typing import NDArray

from .topology  import compute_spectral_embedding, diffusion_distance_matrix, \
                       heat_kernel_signature, hks_cost_matrix
from .biology   import auto_select_radii, neighborhood_distribution_multi_scale, \
                       multi_scale_jsd_cost, normalize_cross_timepoint
from .geometry  import estimate_overlap_fraction, overlap_to_reg_marginals, \
                       recover_rigid_transform, apply_rigid_transform


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(X) -> np.ndarray:
    return X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)


def _cosine_distance_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance (n_A, n_B) float32 — sklearn handles sparse safely."""
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(
        X.astype(np.float32), Y.astype(np.float32)
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_align_orchid(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float,
    # ── Biological cost weights ───────────────────────────────────────────────
    beta: float = 0.3,
    gamma: float = 1.0,
    hks_weight: float = 0.5,
    # ── Algorithmic settings (robust defaults) ────────────────────────────────
    n_spectral: int = 50,
    knn: int = 15,
    n_radii: int = 4,
    diffusion_t: float = 1.0,
    # ── Mode flags ────────────────────────────────────────────────────────────
    cross_timepoint: bool = False,
    use_rep: Optional[str] = None,
    # ── Rigid refinement ─────────────────────────────────────────────────────
    n_refinement_iter: int = 3,
    return_transform: bool = True,
    # ── FUGW solver settings ─────────────────────────────────────────────────
    epsilon: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-7,
    # ── Infrastructure ────────────────────────────────────────────────────────
    filePath: str = './orchid_output',
    sliceA_name: str = 'A',
    sliceB_name: str = 'B',
    overwrite: bool = False,
    verbose: bool = True,
) -> Union[
    NDArray[np.floating],
    Tuple[NDArray[np.floating], np.ndarray, np.ndarray],
]:
    """
    ORCHID pairwise alignment.

    Aligns two spatial transcriptomics slices that may differ in:
    - Rigid body pose (arbitrary rotation + translation)
    - Spatial coverage (partial overlap, unknown extent)
    - Symmetric tissue structures (e.g. bilateral brain, multi-lobe organs)
    - Developmental timepoint (gene expression drift)

    Parameters
    ----------
    sliceA, sliceB : AnnData
        Must contain:
        • .obsm['spatial'] — (n, 2) 2-D spatial coordinates
        • .obs['cell_type_annot'] — cell type string labels
        • .X  or  .obsm[use_rep] — gene expression matrix

    alpha : float ∈ [0, 1]
        Weight of the GW spatial term vs. biological costs.
        0 → pure biology (ignore spatial distances entirely).
        1 → pure spatial (ignore gene expression / cell type).
        Recommended:  0.4–0.6 for same-timepoint;  0.3–0.5 for cross-timepoint.
        Start with 0.5 and adjust based on spatial registration quality.

    beta : float ∈ [0, 1], default 0.3
        Relative weight of cell-type mismatch vs. gene-expression cosine
        distance within the first linear cost term.

    gamma : float, default 1.0
        Weight of the multi-scale neighbourhood JSD term relative to M1.

    hks_weight : float, default 0.5
        Weight of the Heat Kernel Signature cost (symmetry-breaking term).
        Increase toward 1.0 if the tissue has many symmetric regions;
        decrease toward 0.1 for tissues without prominent symmetry.

    n_spectral : int, default 50
        Number of graph Laplacian eigenvectors retained.  50 is sufficient
        for organs with up to ~25 distinct anatomical compartments.

    knn : int, default 15
        k-NN graph degree for graph construction.

    n_radii : int, default 4
        Number of log-spaced radii for multi-scale neighbourhood analysis.

    diffusion_t : float, default 1.0
        Diffusion time for the diffusion distance matrices.  The normalised
        Laplacian makes this dimensionless and robust over [0.5, 3.0].

    cross_timepoint : bool, default False
        Enable intra-cell-type z-score normalisation of gene expression
        before computing the cosine distance.  Use when sliceA and sliceB
        come from different developmental timepoints.

    use_rep : str | None
        obsm key for the expression matrix.  None → use .X.

    n_refinement_iter : int, default 3
        Iterations of the rigid-body refinement loop.  Typically converges
        in 2–3 iterations.

    return_transform : bool, default True
        Return the rigid transform (R, t) in addition to pi.

    epsilon : float, default 0.01
        Entropic regularisation for FUGW.  0 uses the MM solver (exact,
        recommended for datasets up to ~20 k cells).  Set to 0.01–0.1 for
        larger datasets (enables Sinkhorn, requires unbalanced_solver change).

    max_iter, tol : FUGW outer solver budget.

    filePath : str
        Directory for caching intermediate matrices and writing log files.

    sliceA_name, sliceB_name : str
        Labels used in cache filenames and log output.

    overwrite : bool
        Recompute and overwrite all cached files.

    verbose : bool
        Print progress to stdout.

    Returns
    -------
    pi : (n_A, n_B) float64
        Transport plan.  pi[i, j] > 0 indicates cell i in A is matched to
        cell j in B.  pi.sum() < 1 indicates partial overlap was detected.

    R  : (2, 2) float64  rotation matrix          (only if return_transform)
    t  : (2,)   float64  translation vector        (only if return_transform)

    Together (R, t) map B's coordinates to A's frame:
        coords_B_aligned = apply_rigid_transform(sliceB.obsm['spatial'], R, t)

    Notes
    -----
    * All large matrices (D_A, D_B, M_hks, M_nd, M_gene) are float32 to
      halve memory usage vs. float64.  The OT solve uses float64 internally.
    * For n = 15 k cells, peak memory is ~2–3 GB (dominated by n×n matrices).
    * Runtime on CPU: ~5–20 min depending on n and n_refinement_iter.
    """
    t_wall = time.time()
    os.makedirs(filePath, exist_ok=True)

    tag      = f"{sliceA_name}_{sliceB_name}"
    log_path = os.path.join(filePath, f"log_orchid_{tag}.txt")
    _logf    = open(log_path, 'w')

    def _log(msg: str):
        _logf.write(msg + '\n')
        if verbose:
            print(msg)

    _log("=" * 65)
    _log("ORCHID: Orientation-Robust Spatial Transcriptomics Alignment")
    _log(str(datetime.datetime.now()))
    _log(f"  {sliceA_name}: {sliceA.shape[0]} cells × {sliceA.shape[1]} genes")
    _log(f"  {sliceB_name}: {sliceB.shape[0]} cells × {sliceB.shape[1]} genes")
    _log(f"  alpha={alpha}  beta={beta}  gamma={gamma}  hks_weight={hks_weight}")
    _log(f"  knn={knn}  n_spectral={n_spectral}  n_radii={n_radii}")
    _log(f"  cross_timepoint={cross_timepoint}  n_refinement_iter={n_refinement_iter}")
    _log("=" * 65)

    # ══════════════════════════════════════════════════════════════════════════
    # I.  PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[I] Preprocessing")

    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between slices — cannot align.")
    sliceA = sliceA[:, shared_genes].copy()
    sliceB = sliceB[:, shared_genes].copy()
    _log(f"  Shared genes:      {len(shared_genes)}")

    shared_ct = (
        pd.Index(sliceA.obs['cell_type_annot']).unique()
        .intersection(pd.Index(sliceB.obs['cell_type_annot']).unique())
    )
    if len(shared_ct) == 0:
        raise ValueError("No shared cell types between slices — cannot align.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_ct)].copy()
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_ct)].copy()
    _log(f"  Shared cell types: {len(shared_ct)}")
    _log(f"  Post-filter sizes: A={sliceA.shape[0]}, B={sliceB.shape[0]}")

    # Canonical cell-type ordering (must be identical for both slices)
    cell_type_order: List[str] = sorted(shared_ct.tolist())

    if cross_timepoint:
        _log("  Applying intra-cell-type z-score normalization …")
        sliceA, sliceB = normalize_cross_timepoint(sliceA, sliceB, use_rep=use_rep)

    coords_A = sliceA.obsm['spatial'].copy().astype(np.float64)
    coords_B = sliceB.obsm['spatial'].copy().astype(np.float64)
    n_A, n_B = sliceA.shape[0], sliceB.shape[0]

    # ══════════════════════════════════════════════════════════════════════════
    # II.  SPECTRAL EMBEDDINGS  (shared computation for D and HKS)
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[II] Spectral embeddings  (rotation-invariant graph Laplacian)")

    def _get_spectral(name: str, coords: np.ndarray):
        cache = os.path.join(filePath, f"spec_{name}.npz")
        if os.path.exists(cache) and not overwrite:
            _log(f"  Loading cached spectral embedding [{name}]")
            z = np.load(cache)
            return z['eigenvalues'], z['eigenvectors']
        _log(f"  Computing spectral embedding [{name}]  (n={coords.shape[0]}) …")
        vals, vecs = compute_spectral_embedding(coords, k=knn, n_components=n_spectral)
        np.savez(cache, eigenvalues=vals, eigenvectors=vecs)
        return vals, vecs

    evals_A, evecs_A = _get_spectral(sliceA_name, coords_A)
    evals_B, evecs_B = _get_spectral(sliceB_name, coords_B)

    _log(f"  Spectral range A: [{float(evals_A.min()):.4f}, {float(evals_A.max()):.4f}]")
    _log(f"  Spectral range B: [{float(evals_B.min()):.4f}, {float(evals_B.max()):.4f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # III. DIFFUSION DISTANCE MATRICES  (GW cost matrices)
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[III] Diffusion distance matrices  (t={})".format(diffusion_t))

    def _get_diff_dist(name: str, coords: np.ndarray,
                       evals: np.ndarray, evecs: np.ndarray) -> np.ndarray:
        cache = os.path.join(filePath, f"diffdist_{name}.npy")
        if os.path.exists(cache) and not overwrite:
            _log(f"  Loading cached diffusion distances [{name}]")
            return np.load(cache)
        _log(f"  Computing diffusion distances [{name}] …")
        D = diffusion_distance_matrix(
            coords, k=knn, n_components=n_spectral,
            t=diffusion_t, eigenvalues=evals, eigenvectors=evecs,
        )
        np.save(cache, D)
        return D

    D_A = _get_diff_dist(sliceA_name, coords_A, evals_A, evecs_A)
    D_B = _get_diff_dist(sliceB_name, coords_B, evals_B, evecs_B)

    _log(f"  D_A: {D_A.shape}  range=[{D_A.min():.4f}, {D_A.max():.4f}]")
    _log(f"  D_B: {D_B.shape}  range=[{D_B.min():.4f}, {D_B.max():.4f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # IV.  HEAT KERNEL SIGNATURE COST  (symmetry-breaking)
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[IV] Heat Kernel Signature cost  (symmetry breaking)")

    def _get_hks(name: str, coords: np.ndarray,
                 evals: np.ndarray, evecs: np.ndarray) -> np.ndarray:
        cache = os.path.join(filePath, f"hks_{name}.npy")
        if os.path.exists(cache) and not overwrite:
            _log(f"  Loading cached HKS [{name}]")
            return np.load(cache)
        _log(f"  Computing HKS [{name}] …")
        H = heat_kernel_signature(
            coords, k=knn, n_components=n_spectral,
            eigenvalues=evals, eigenvectors=evecs,
        )
        np.save(cache, H)
        return H

    hks_A = _get_hks(sliceA_name, coords_A, evals_A, evecs_A)
    hks_B = _get_hks(sliceB_name, coords_B, evals_B, evecs_B)

    cache_M_hks = os.path.join(filePath, f"M_hks_{tag}.npy")
    if os.path.exists(cache_M_hks) and not overwrite:
        M_hks = np.load(cache_M_hks)
        _log("  Loading cached M_hks")
    else:
        _log("  Computing HKS cost matrix …")
        M_hks = hks_cost_matrix(hks_A, hks_B)
        np.save(cache_M_hks, M_hks)
    _log(f"  M_hks: {M_hks.shape}  range=[{M_hks.min():.4f}, {M_hks.max():.4f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # V.   BIOLOGICAL LINEAR COST
    #      M_linear = (1−β)·M_gene + β·M_ct + γ·M_nd + ω·M_hks
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[V] Biological linear cost")

    # — Gene expression (cosine distance) ——————————————————————————————————
    cache_M_gene = os.path.join(filePath, f"cosine_{tag}.npy")
    if os.path.exists(cache_M_gene) and not overwrite:
        M_gene = np.load(cache_M_gene)
        _log("  Loading cached cosine distance (gene expression)")
    else:
        _log("  Computing cosine distance (gene expression) …")
        X_A = _to_dense(sliceA.X if use_rep is None else sliceA.obsm[use_rep])
        X_B = _to_dense(sliceB.X if use_rep is None else sliceB.obsm[use_rep])
        X_A = X_A.astype(np.float32) + 0.01
        X_B = X_B.astype(np.float32) + 0.01
        M_gene = _cosine_distance_matrix(X_A, X_B)
        np.save(cache_M_gene, M_gene)

    # — Cell-type mismatch ————————————————————————————————————————————————
    lab_A = np.array(sliceA.obs['cell_type_annot'].values)
    lab_B = np.array(sliceB.obs['cell_type_annot'].values)
    M_ct  = (lab_A[:, None] != lab_B[None, :]).astype(np.float32)

    # — Multi-scale neighbourhood JSD ————————————————————————————————————
    radii_A = auto_select_radii(coords_A, n_radii=n_radii)
    radii_B = auto_select_radii(coords_B, n_radii=n_radii)
    # Geometric mean of the two slices' natural radii (handles size mismatch)
    radii = np.sqrt(radii_A * radii_B)
    _log(f"  Multi-scale radii: {[f'{r:.1f}' for r in radii.tolist()]}")

    cache_nd_A = os.path.join(filePath, f"nd_ms_{sliceA_name}.npy")
    cache_nd_B = os.path.join(filePath, f"nd_ms_{sliceB_name}.npy")
    cache_M_nd = os.path.join(filePath, f"M_nd_{tag}.npy")

    if os.path.exists(cache_nd_A) and not overwrite:
        nd_A = np.load(cache_nd_A)
    else:
        _log(f"  Computing multi-scale ND [{sliceA_name}] …")
        nd_A = neighborhood_distribution_multi_scale(
            sliceA, radii=radii, cell_type_order=cell_type_order)
        np.save(cache_nd_A, nd_A)

    if os.path.exists(cache_nd_B) and not overwrite:
        nd_B = np.load(cache_nd_B)
    else:
        _log(f"  Computing multi-scale ND [{sliceB_name}] …")
        nd_B = neighborhood_distribution_multi_scale(
            sliceB, radii=radii, cell_type_order=cell_type_order)
        np.save(cache_nd_B, nd_B)

    if os.path.exists(cache_M_nd) and not overwrite:
        M_nd = np.load(cache_M_nd)
        _log("  Loading cached multi-scale JSD cost")
    else:
        _log("  Computing multi-scale JSD cost matrix …")
        M_nd = multi_scale_jsd_cost(nd_A, nd_B)
        np.save(cache_M_nd, M_nd)
    _log(f"  M_nd: {M_nd.shape}  range=[{M_nd.min():.4f}, {M_nd.max():.4f}]")

    # — Combined cost ——————————————————————————————————————————————————————
    M1      = (1.0 - beta) * M_gene + beta * M_ct         # (n_A, n_B) float32
    M_bio   = (M1 + gamma * M_nd + hks_weight * M_hks).astype(np.float64)
    _log(f"  M_bio range: [{M_bio.min():.4f}, {M_bio.max():.4f}]")

    # ══════════════════════════════════════════════════════════════════════════
    # VI.  ADAPTIVE FUGW SOLVE
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[VI] Fused Unbalanced Gromov-Wasserstein solve")

    overlap_frac  = estimate_overlap_fraction(sliceA, sliceB)
    reg_marginals = overlap_to_reg_marginals(overlap_frac, M_bio)
    _log(f"  Estimated overlap fraction: {overlap_frac:.3f}")
    _log(f"  Derived reg_marginals:      {reg_marginals:.5f}")

    # FUGW uses a different convention for the biology/space trade-off:
    #   balanced FGW:  (1−α)·<M_bio,π>  +  α·GW(D_A,D_B,π)
    #   FUGW (POT):    GW  +  α_fugw·<M_bio,π>  +  unbalanced penalty
    # Matching the ratio: α_fugw = (1−α)/α
    alpha_fugw = (1.0 - alpha) / (alpha + 1e-10)
    _log(f"  alpha={alpha} → alpha_fugw={alpha_fugw:.4f}")

    a = np.ones(n_A, dtype=np.float64) / n_A
    b = np.ones(n_B, dtype=np.float64) / n_B

    pi, _, log_dict = ot.gromov.fused_unbalanced_gromov_wasserstein(
        Cx=D_A.astype(np.float64),
        Cy=D_B.astype(np.float64),
        wx=a,
        wy=b,
        reg_marginals=reg_marginals,
        epsilon=epsilon,
        divergence='kl',
        unbalanced_solver='sinkhorn',
        alpha=alpha_fugw,
        M=M_bio,
        init_pi=None,
        max_iter=max_iter,
        tol=tol,
        log=True,
        verbose=False,
    )

    pi = np.asarray(pi, dtype=np.float64)
    pi_mass = float(pi.sum())
    _log(f"  Initial pi mass: {pi_mass:.4f}  (< 1 → partial overlap detected)")

    # ══════════════════════════════════════════════════════════════════════════
    # VII. ITERATIVE RIGID REFINEMENT
    #      Each iteration: recover (R, t) → transform B → recompute D_B → re-solve
    #
    #  Rationale: the diffusion distances are topology-invariant within each
    #  slice, but after estimating the initial plan we can recover an explicit
    #  Euclidean transform, re-embed B in A's coordinate frame, recompute D_B
    #  from the aligned coordinates, and re-solve.  After alignment, the two
    #  slices share the same spatial frame, so D_A and D_B should become more
    #  compatible, improving the GW matching in subsequent iterations.
    # ══════════════════════════════════════════════════════════════════════════

    _log("\n[VII] Iterative rigid refinement")

    # Accumulated rigid transform  (B → A frame)
    R_acc = np.eye(2,  dtype=np.float64)
    t_acc = np.zeros(2, dtype=np.float64)
    coords_B_cur = coords_B.copy()

    for it in range(n_refinement_iter):
        _log(f"  Iteration {it + 1}/{n_refinement_iter}")

        # (a) Recover incremental rigid transform from current plan
        R_it, t_it = recover_rigid_transform(coords_A, coords_B_cur, pi)

        # (b) Compose with accumulated transform
        t_acc = R_it @ t_acc + t_it
        R_acc = R_it @ R_acc

        # (c) Apply to original B coordinates (avoids error accumulation)
        coords_B_cur = apply_rigid_transform(coords_B, R_acc, t_acc)

        # (d) Recompute D_B from aligned coordinates
        #     Spectral embedding changes with coordinates, so we recompute.
        _log(f"    Recomputing D_B from aligned coordinates …")
        evals_B_it, evecs_B_it = compute_spectral_embedding(
            coords_B_cur, k=knn, n_components=n_spectral,
        )
        D_B_it = diffusion_distance_matrix(
            coords_B_cur, k=knn, n_components=n_spectral,
            t=diffusion_t, eigenvalues=evals_B_it, eigenvectors=evecs_B_it,
        )

        # (e) Re-solve with previous plan as warm start
        pi_prev = pi.copy()
        pi, _, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
            Cx=D_A.astype(np.float64),
            Cy=D_B_it.astype(np.float64),
            wx=a,
            wy=b,
            reg_marginals=reg_marginals,
            epsilon=epsilon,
            divergence='kl',
            unbalanced_solver='mm',
            alpha=alpha_fugw,
            M=M_bio,
            init_pi=pi_prev,
            max_iter=max_iter,
            tol=tol,
            log=True,
            verbose=False,
        )
        pi = np.asarray(pi, dtype=np.float64)

        delta = float(np.abs(pi - pi_prev).mean())
        _log(f"    Plan delta (mean |Δπ|): {delta:.6f}  pi_mass={pi.sum():.4f}")

        if delta < 1e-5:
            _log("    Converged early.")
            break

    # ══════════════════════════════════════════════════════════════════════════
    # Logging & return
    # ══════════════════════════════════════════════════════════════════════════

    runtime = time.time() - t_wall
    _log(f"\nTotal runtime: {runtime:.1f} s")
    _log(f"Final pi mass: {float(pi.sum()):.4f}")
    _log(f"Rigid transform R:\n{R_acc}")
    _log(f"Rigid transform t: {t_acc}")
    _logf.close()

    # Save final plan and transform
    np.save(os.path.join(filePath, f"pi_{tag}.npy"), pi)
    np.savez(
        os.path.join(filePath, f"rigid_{tag}.npz"),
        R=R_acc, t=t_acc,
    )

    if return_transform:
        return pi, R_acc, t_acc
    return pi
