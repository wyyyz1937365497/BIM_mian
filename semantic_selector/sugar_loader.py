"""Load SuGaR / 3DGS Gaussian models.

This module dynamically adds the GaussianEditor and SuGaR package roots to
``sys.path`` so that the downstream code can import from
``gaussiansplatting`` / ``sugar_*`` without additional setup.
"""

import glob
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — point to the GaussianEditor and SuGaR source trees.
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
_SUGAR_ROOT = os.path.join(_PROJECT_ROOT, "examples", "SuGaR")

for _p in (
    _GAUSSIAN_EDITOR_ROOT,
    _GAUSSIANSPLATTING_ROOT,
    _DIFF_GAUSSIAN_RASTERIZATION_ROOT,
    _SUGAR_ROOT,
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


# ---------------------------------------------------------------------------
# SuGaR Refine
# ---------------------------------------------------------------------------

def refine_gaussian_model(
    scene_path: str,
    checkpoint_path: str,
    mesh_path: str,
    output_dir: str = None,
    iteration_to_load: int = 7000,
    refinement_iterations: int = 15_000,
    gaussians_per_triangle: int = 1,
    n_vertices_in_fg: int = 1_000_000,
    normal_consistency_factor: float = 0.1,
    bboxmin: str = None,
    bboxmax: str = None,
    eval: bool = True,
    white_background: bool = False,
    export_ply: bool = True,
    gpu: int = 0,
) -> str:
    """Run SuGaR refinement on a vanilla 3DGS checkpoint and return the
    path to the refined ``.ply`` file.

    This is a thin wrapper around
    ``sugar_trainers.refine.refined_training`` that mimics the argument
    namespace expected by that function.

    Parameters
    ----------
    scene_path : str
        Path to the COLMAP / NeRF dataset directory (must contain
        ``sparse/`` or ``images/``).
    checkpoint_path : str
        Path to the vanilla 3DGS checkpoint directory (the one that
        contains ``point_cloud/iteration_*/point_cloud.ply``).
    mesh_path : str
        Path to the extracted surface mesh (``.ply``) produced by the
        coarse SuGaR stage.
    output_dir : str, optional
        Where to write SuGaR outputs.  Defaults to
        ``./output/refined/<scene_name>``.
    iteration_to_load : int
        Which 3DGS iteration to load (default 7000).
    refinement_iterations : int
        Number of refine optimisation steps (default 15 000).
    gaussians_per_triangle : int
        Gaussians placed on each mesh triangle (default 1).
    n_vertices_in_fg : int
        Target vertex count for the foreground mesh (default 1 000 000).
    normal_consistency_factor : float
        Weight for the normal-consistency regulariser (default 0.1).
    bboxmin, bboxmax : str, optional
        Custom foreground bounding box, e.g. ``"(0,0,0)"``.
    eval : bool
        Use the eval split of the dataset.
    white_background : bool
        Optimise with a white background.
    export_ply : bool
        Export a refined ``.ply`` after training (default True).
    gpu : int
        CUDA device index.

    Returns
    -------
    str
        Absolute path to the exported refined ``.ply`` file.

    Raises
    ------
    FileNotFoundError
        If the refined ``.ply`` was not found after training (only when
        *export_ply* is True).
    """
    from sugar_trainers.refine import refined_training

    class _RefineArgs:
        """Minimal namespace matching what ``refined_training`` reads."""
        pass

    args = _RefineArgs()
    args.scene_path = scene_path
    args.checkpoint_path = checkpoint_path
    args.mesh_path = mesh_path
    args.output_dir = output_dir
    args.iteration_to_load = iteration_to_load
    args.refinement_iterations = refinement_iterations
    args.gaussians_per_triangle = gaussians_per_triangle
    args.n_vertices_in_fg = n_vertices_in_fg
    args.normal_consistency_factor = normal_consistency_factor
    args.bboxmin = bboxmin
    args.bboxmax = bboxmax
    args.eval = eval
    args.white_background = white_background
    args.export_ply = export_ply
    args.gpu = gpu

    # --- Run the refinement (may take a long time) -----------------------
    model_path = refined_training(args)

    if not export_ply:
        return model_path

    # --- Locate the exported .ply ----------------------------------------
    # refined_training builds the ply path from model_path by replacing
    # the 4th-from-last path component with 'refined_ply' and appending
    # '.ply'.  We mirror that logic here.
    tmp = model_path.rstrip(os.sep).split(os.sep)
    if len(tmp) >= 4:
        tmp[-4] = "refined_ply"
        tmp.pop(-1)
        tmp[-1] = tmp[-1] + ".ply"
        candidate = os.path.join(*tmp)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # Fallback: search for any .ply under the output directory.
    search_root = output_dir or os.path.join("./output/refined", os.path.basename(scene_path))
    candidates = sorted(glob.glob(os.path.join(search_root, "**", "*.ply"), recursive=True))
    if candidates:
        return os.path.abspath(candidates[-1])

    raise FileNotFoundError(
        f"SuGaR refinement completed but no .ply was found under "
        f"{search_root}.  Set export_ply=True and check the output."
    )
