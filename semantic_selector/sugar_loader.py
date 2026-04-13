"""Load SuGaR / 3DGS Gaussian models.

This module dynamically adds the GaussianEditor package roots to
``sys.path`` so that the downstream code can import from
``gaussiansplatting`` without additional setup.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup — point to the GaussianEditor source trees.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_GAUSSIAN_EDITOR_ROOT = os.path.join(
    _PROJECT_ROOT, "examples", "GaussianEditor"
)
_GAUSSIANSPLATTING_ROOT = os.path.join(
    _GAUSSIAN_EDITOR_ROOT, "gaussiansplatting"
)
_DIFF_GAUSSIAN_RASTERIZATION_ROOT = os.path.join(
    _GAUSSIAN_EDITOR_ROOT,
    "gaussiansplatting",
    "submodules",
    "diff-gaussian-rasterization",
)

for _p in (
    _GAUSSIAN_EDITOR_ROOT,
    _GAUSSIANSPLATTING_ROOT,
    _DIFF_GAUSSIAN_RASTERIZATION_ROOT,
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_gaussian_model(model_path: str, sh_degree: int = 3):
    """Load a Gaussian model from a ``.ply`` file.

    Parameters
    ----------
    model_path : str
        Path to a ``.ply`` file produced by 3DGS training or SuGaR refine.
    sh_degree : int
        Maximum spherical-harmonics degree (default 3).

    Returns
    -------
    gaussiansplatting.scene.gaussian_model.GaussianModel
        Model instance with weights loaded and ``localize == False``.

    Notes
    -----
    TODO: SuGaR ``.pth`` format support will be added once the SuGaR
    integration is complete.
    """
    from gaussiansplatting.scene.gaussian_model import GaussianModel

    model = GaussianModel(sh_degree=sh_degree, anchor_weight_init_g0=0,
                          anchor_weight_init=0, anchor_weight_multiplier=0)
    model.load_ply(model_path)

    # Ensure we operate on the full set of Gaussians.
    model.localize = False

    return model


def load_gaussians_xyz(model) -> "np.ndarray":
    """Return Gaussian 3-D positions as a NumPy array.

    Parameters
    ----------
    model : GaussianModel

    Returns
    -------
    np.ndarray
        Shape ``(N, 3)``, dtype float32.
    """
    import numpy as np

    return model._xyz.detach().cpu().numpy()
