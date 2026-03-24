"""
orchid/geometry.py
==================
Geometric utilities for the ORCHID alignment framework.

Three functions:

1. **Overlap fraction estimation**
   Given partial, differently-cut slices, we never know a priori what
   fraction of cells have true spatial counterparts in the other slice.
   Setting the unbalanced-OT regularisation parameter reg_marginals
   requires an estimate of this fraction.

   We estimate it from the *cell-type histogram overlap* — the summed
   minimum frequency across shared cell types:

       f = Σ_c  min( p_A(c), p_B(c) )

   Rationale: if region X is present in A but absent in B, the cell types
   characteristic of X will appear in A but not B, reducing the histogram
   overlap.  Conversely, full spatial overlap gives identical histograms,
   so f → 1.  This is a hyperparameter-free, biologically grounded proxy.

2. **reg_marginals derivation**
   Maps the overlap estimate f ∈ (0,1] to the FUGW reg_marginals parameter
   using a physically motivated formula that ensures:
   - f → 1 (full overlap)  → large reg_marginals (≈ balanced OT)
   - f → 0 (no overlap)    → small reg_marginals (all mass destroyable)

3. **Rigid body recovery (weighted Procrustes)**
   Given a transport plan π, recovers the rigid transform (R, t) that maps
   slice B's coordinates closest to slice A's.  Uses weighted Procrustes
   analysis (Umeyama, 1991), with weights derived from the transport plan's
   row/column marginals.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Overlap estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_overlap_fraction(
    sliceA: AnnData,
    sliceB: AnnData,
) -> float:
    """
    Estimate the fraction of cells expected to have spatial counterparts.

    f = Σ_c  min( p_A(c), p_B(c) )

    where p_A, p_B are normalised cell-type frequency histograms.  This
    equals 1 minus half the L1 (total variation) distance between the
    histograms, and is bounded in [0, 1].

    Edge case handling:
    - Cell types absent in one slice contribute 0 (no match possible)
    - Result is clipped to [0.05, 1.0] to avoid degenerate reg_marginals

    Parameters
    ----------
    sliceA, sliceB : AnnData with .obs['cell_type_annot']

    Returns
    -------
    f : float — estimated overlap fraction in [0.05, 1.0]
    """
    ct_A = sliceA.obs['cell_type_annot'].value_counts(normalize=True)
    ct_B = sliceB.obs['cell_type_annot'].value_counts(normalize=True)

    all_ct = ct_A.index.union(ct_B.index)
    overlap = sum(
        min(float(ct_A.get(c, 0.0)), float(ct_B.get(c, 0.0)))
        for c in all_ct
    )
    return float(np.clip(overlap, 0.05, 1.0))


def overlap_to_reg_marginals(
    f: float,
    M_bio: np.ndarray,
) -> float:
    """
    Derive the FUGW reg_marginals from the estimated overlap fraction.

    Formula:

        reg_marginals = [ f / (1 − f + ε) ] × median_cost

    where median_cost is the median entry of the combined biological cost
    matrix.  The factor f/(1−f+ε) is the odds ratio of the overlap fraction,
    which maps:
        f = 1.0  →  reg_marginals = (1/ε) × cost  (nearly balanced)
        f = 0.5  →  reg_marginals = 1.0  × cost  (symmetric penalty)
        f = 0.05 →  reg_marginals ≈ 0.05 × cost  (strongly unbalanced)

    Multiplying by the median cost makes the penalty scale-invariant: the
    same degree of imbalance tolerance is enforced regardless of the
    absolute magnitude of the cost matrix.

    Result is clipped to [0.01, 100] × median_cost for numerical safety.

    Parameters
    ----------
    f      : overlap fraction from ``estimate_overlap_fraction``
    M_bio  : (n_A, n_B) combined biological cost matrix

    Returns
    -------
    reg_m : float
    """
    median_cost = float(np.median(M_bio))
    if median_cost < 1e-10:
        median_cost = 1e-2          # fallback for pathological inputs

    odds  = f / (1.0 - f + 0.05)
    reg_m = odds * median_cost

    return float(np.clip(reg_m, 0.01 * median_cost, 100.0 * median_cost))


# ─────────────────────────────────────────────────────────────────────────────
# Rigid body recovery
# ─────────────────────────────────────────────────────────────────────────────

def recover_rigid_transform(
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover the rigid body transformation (R, t) from a transport plan.

    Solves the weighted Procrustes problem:

        min_{R ∈ SO(d), t ∈ ℝᵈ}  Σ_{ij} π_{ij} ‖x^A_i − (R x^B_j + t)‖²

    Closed-form solution (Umeyama, 1991):
        1. Compute π-weighted centroids μ_A, μ_B
        2. Form the cross-covariance  H = (Y − μ_B)ᵀ Π̃ᵀ (X − μ_A)
           where Π̃ is the plan normalised to unit mass
        3. Decompose H = U Σ Vᵀ  via SVD
        4. R = Vᵀ D Uᵀ  where D = diag(1, …, 1, det(Vᵀ Uᵀ)) ensures det=+1
        5. t = μ_A − R μ_B

    Works with unbalanced plans (π.sum() < 1): the plan is normalised
    internally so that only the *relative* weights matter, not the total mass.

    Parameters
    ----------
    coords_A : (n_A, d) spatial coordinates of slice A (target frame)
    coords_B : (n_B, d) spatial coordinates of slice B (source to transform)
    pi       : (n_A, n_B) transport plan (may be unbalanced)

    Returns
    -------
    R : (d, d) rotation matrix with det(R) = +1
    t : (d,)   translation vector

    Usage:  coords_B_aligned = apply_rigid_transform(coords_B, R, t)
    """
    total_mass = float(pi.sum())
    if total_mass < 1e-12:
        d = coords_A.shape[1]
        return np.eye(d, dtype=np.float64), np.zeros(d, dtype=np.float64)

    pi_norm = pi.astype(np.float64) / total_mass      # normalise to unit mass

    # π-weighted centroids  (marginals × coords)
    w_A = pi_norm.sum(axis=1)      # (n_A,)
    w_B = pi_norm.sum(axis=0)      # (n_B,)

    mu_A = (w_A[:, None] * coords_A.astype(np.float64)).sum(axis=0)
    mu_B = (w_B[:, None] * coords_B.astype(np.float64)).sum(axis=0)

    # Centred coordinates
    X = coords_A.astype(np.float64) - mu_A[None, :]   # (n_A, d)
    Y = coords_B.astype(np.float64) - mu_B[None, :]   # (n_B, d)

    # Weighted cross-covariance:  H = Yᵀ Π̃ᵀ X    shape (d, d)
    H = Y.T @ pi_norm.T @ X

    # SVD
    U, _, Vt = np.linalg.svd(H)

    # Enforce proper rotation  (det = +1, not a reflection)
    d = coords_A.shape[1]
    sign_correction = np.ones(d)
    sign_correction[-1] = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag(sign_correction) @ U.T         # (d, d)

    t = mu_A - R @ mu_B                                # (d,)

    return R.astype(np.float64), t.astype(np.float64)


def apply_rigid_transform(
    coords: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Apply rigid transform to a coordinate array.

    coords_new[i] = R @ coords[i] + t

    Parameters
    ----------
    coords : (n, d) coordinate array
    R      : (d, d) rotation matrix
    t      : (d,) translation vector

    Returns
    -------
    (n, d) transformed coordinates, same dtype as input
    """
    dtype = coords.dtype
    return (coords.astype(np.float64) @ R.T + t[None, :]).astype(dtype)
