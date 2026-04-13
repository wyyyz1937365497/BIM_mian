"""Interactive Semantic Selector for 3DGS / SuGaR pipelines.

Selects specific Gaussians via multi-view 2D masks (from VLM+SAM3),
using the GaussianEditor apply_weights CUDA kernel.
"""

from semantic_selector.core import InteractiveSemanticSelector
from semantic_selector.camera_utils import (
    create_orbital_cameras,
    compute_gaussian_centroid,
    estimate_gaussian_extent,
)
from semantic_selector.mesh_transfer import transfer_labels_to_mesh, save_colored_mesh
from semantic_selector.sugar_loader import load_gaussian_model, load_gaussians_xyz

__all__ = [
    "InteractiveSemanticSelector",
    "create_orbital_cameras",
    "compute_gaussian_centroid",
    "estimate_gaussian_extent",
    "transfer_labels_to_mesh",
    "save_colored_mesh",
    "load_gaussian_model",
    "load_gaussians_xyz",
]
