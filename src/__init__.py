"""
ORCHID: Orientation-Robust, Cross-resolution, Hierarchical Integration
        for spatial transcriptomics Data

Extends the INCENT FGW framework with six innovations that together enable
correct alignment under conditions where the original INCENT fails:

  1. Rotation / translation invariance
     Diffusion distances replace Euclidean spatial distances as the GW
     intra-slice structure matrices.  Computed via the normalised graph
     Laplacian, they depend only on tissue topology, not absolute pose.

  2. Scale invariance
     The normalised Laplacian normalises by vertex degree, making spectral
     features independent of the tissue's physical extent.

  3. Multi-scale biological cost
     Neighbourhood cell-type distributions are computed at log-spaced radii
     spanning the data's own spatial scale (intercellular → mesoscale).
     Radii are chosen automatically — no hyperparameter.

  4. Topological symmetry disambiguation
     Heat Kernel Signatures (HKS) give each cell a global topological
     fingerprint.  Even in bilaterally- or n-fold-symmetric organs, partial
     slices carry different global topologies, making HKS discriminative.
     Used as an additive linear cost term in the OT objective.

  5. Adaptive partial-overlap handling
     The overlap fraction is estimated from cell-type histogram matching and
     converted to the FUGW reg_marginals parameter, eliminating a key
     hyperparameter of unbalanced OT.

  6. Cross-timepoint normalization
     Intra-cell-type z-scoring removes temporal expression drift while
     preserving spatially-informative within-type variation.

Minimal user-facing API
-----------------------
    from orchid import pairwise_align_orchid, apply_rigid_transform

    pi, R, t = pairwise_align_orchid(sliceA, sliceB, alpha=0.5)

    # Align B's coordinates to A's frame
    from orchid import apply_rigid_transform
    coords_B_aligned = apply_rigid_transform(sliceB.obsm['spatial'], R, t)
"""

from .align    import pairwise_align_orchid
from .topology import (
    compute_spectral_embedding,
    diffusion_distance_matrix,
    heat_kernel_signature,
    hks_cost_matrix,
)
from .biology  import (
    auto_select_radii,
    neighborhood_distribution_multi_scale,
    multi_scale_jsd_cost,
    normalize_cross_timepoint,
)
from .geometry import (
    estimate_overlap_fraction,
    overlap_to_reg_marginals,
    recover_rigid_transform,
    apply_rigid_transform,
)

__all__ = [
    # Primary entry point
    'pairwise_align_orchid',
    # Topology
    'compute_spectral_embedding',
    'diffusion_distance_matrix',
    'heat_kernel_signature',
    'hks_cost_matrix',
    # Biology
    'auto_select_radii',
    'neighborhood_distribution_multi_scale',
    'multi_scale_jsd_cost',
    'normalize_cross_timepoint',
    # Geometry
    'estimate_overlap_fraction',
    'overlap_to_reg_marginals',
    'recover_rigid_transform',
    'apply_rigid_transform',
]

__version__ = '1.0.0'
