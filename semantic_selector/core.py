"""Core Interactive Semantic Selector.

Leverages the GaussianEditor ``apply_weights`` CUDA kernel to identify
which Gaussians contribute to a given 2-D mask region across one or more
camera views.
"""

import numpy as np
import torch

from gaussiansplatting.gaussian_renderer import camera2rasterizer, render


class InteractiveSemanticSelector:
    """Select Gaussians from multi-view 2-D masks.

    Parameters
    ----------
    gaussian_model : gaussiansplatting.scene.gaussian_model.GaussianModel
        A loaded Gaussian model (**not** in ``localize`` mode).
    image_height, image_width : int
        Default rendering resolution.
    """

    def __init__(self, gaussian_model, image_height: int = 512,
                 image_width: int = 512):
        self.gaussian_model = gaussian_model
        self.image_height = image_height
        self.image_width = image_width

        if getattr(gaussian_model, "localize", False):
            raise ValueError(
                "gaussian_model.localize is True — selection requires the "
                "full set of Gaussians.  Set model.localize = False first."
            )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def render_views_for_selection(self, cameras):
        """Render RGB + depth for each camera (for visual preview).

        Parameters
        ----------
        cameras : list
            Camera objects accepted by ``gaussian_renderer.render``.

        Returns
        -------
        list[dict]
            Each dict has keys ``"render"`` (RGB), ``"depth_3dgs"``,
            ``"visibility_filter"``.
        """
        bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32,
                                device="cuda")
        # Minimal pipe namespace — render() only reads .compute_cov3D_python
        # and .convert_SHs_python from it.
        from types import SimpleNamespace
        pipe = SimpleNamespace(compute_cov3D_python=False,
                               convert_SHs_python=False)

        results = []
        for cam in cameras:
            out = render(cam, self.gaussian_model, pipe, bg_color)
            results.append(out)
        return results

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def extract_gids_from_mask(self, camera, mask: np.ndarray):
        """Identify Gaussians that contribute to pixels inside *mask*.

        Uses ``GaussianRasterizer.apply_weights`` so that every Gaussian
        whose splat overlaps a foreground (non-zero) pixel is counted.

        Parameters
        ----------
        camera : Camera-like
            Any camera object compatible with ``camera2rasterizer``.
        mask : np.ndarray
            Binary or soft mask, shape ``(H, W)``, dtype ``uint8`` or
            ``float32``.  Non-zero pixels are "foreground".

        Returns
        -------
        set[int]
            Indices of the selected Gaussians.
        """
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(
                mask.astype(np.float32)
            ).unsqueeze(0).cuda()  # (1, H, W)
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.float().unsqueeze(0).cuda()
        else:
            raise TypeError(f"mask must be np.ndarray or torch.Tensor, got {type(mask)}")

        N = self.gaussian_model.get_xyz.shape[0]
        weights = torch.zeros((N, 1), dtype=torch.float32, device="cuda")
        weights_cnt = torch.zeros((N, 1), dtype=torch.int32, device="cuda")

        bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32,
                                device="cuda")
        rasterizer = camera2rasterizer(camera, bg_color, sh_degree=0)

        rasterizer.apply_weights(
            means3D=self.gaussian_model.get_xyz,
            means2D=None,
            opacities=self.gaussian_model.get_opacity,
            shs=None,
            weights=weights,
            scales=self.gaussian_model.get_scaling,
            rotations=self.gaussian_model.get_rotation,
            cov3Ds_precomp=None,
            cnt=weights_cnt,
            image_weights=mask_tensor,
        )

        # cnt > 0  ⟹  the Gaussian contributed to ≥1 foreground pixel.
        selected_indices = (weights_cnt > 0).squeeze(1).nonzero(
            as_tuple=True
        )[0]
        return set(selected_indices.cpu().tolist())

    # ------------------------------------------------------------------
    # Multi-view merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge_multiview_gids(gid_sets):
        """Merge Gaussian-ID sets from multiple views (union).

        Parameters
        ----------
        gid_sets : list[set[int]]

        Returns
        -------
        set[int]
        """
        merged: set = set()
        for s in gid_sets:
            merged.update(s)
        return merged
