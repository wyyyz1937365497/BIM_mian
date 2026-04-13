"""Virtual camera generation for multi-view Gaussian selection.

Uses C2W_Camera from gaussiansplatting to create orbital camera trajectories.
"""

import math
import numpy as np
import torch


def create_orbital_cameras(
    center: np.ndarray,
    radius: float,
    n_cameras: int = 10,
    elevation_deg: float = 30.0,
    image_height: int = 512,
    image_width: int = 512,
    fov_y: float = 0.85,
):
    """Create cameras arranged in an orbital ring around *center*.

    Parameters
    ----------
    center : np.ndarray
        World-space point to orbit around, shape ``(3,)``.
    radius : float
        Distance from *center* to each camera.
    n_cameras : int
        Number of viewpoints.
    elevation_deg : float
        Elevation angle in degrees above the horizontal plane.
    image_height, image_width : int
        Render resolution.
    fov_y : float
        Vertical field-of-view in **radians**.

    Returns
    -------
    list[C2W_Camera]
    """
    # Deferred import so the module can be inspected without CUDA.
    from gaussiansplatting.scene.cameras import C2W_Camera

    center = np.asarray(center, dtype=np.float64)
    elevation = math.radians(elevation_deg)
    cameras = []

    for i in range(n_cameras):
        azimuth = 2.0 * math.pi * i / n_cameras

        # Camera position in world space.
        eye = center + radius * np.array([
            math.cos(azimuth) * math.cos(elevation),
            math.sin(elevation),
            math.sin(azimuth) * math.cos(elevation),
        ])

        # Build c2w (camera-to-world) matrix — OpenGL convention, -Z forward.
        forward = center - eye
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            # Degenerate case: looking straight up/down.
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        c2w = torch.zeros(4, 4, dtype=torch.float32)
        c2w[0, :3] = torch.from_numpy(right)
        c2w[1, :3] = torch.from_numpy(up)
        c2w[2, :3] = torch.from_numpy(-forward)  # -Z forward
        c2w[:3, 3] = torch.from_numpy(eye).float()
        c2w[3, 3] = 1.0

        cam = C2W_Camera(
            c2w=c2w,
            FoVy=fov_y,
            height=image_height,
            width=image_width,
            azimuth=azimuth,
            elevation=elevation_deg,
            dist=radius,
        )
        cameras.append(cam)

    return cameras


def compute_gaussian_centroid(gaussian_xyz, opacity=None):
    """Compute the centroid of Gaussian point cloud.

    Parameters
    ----------
    gaussian_xyz : torch.Tensor or np.ndarray
        Shape ``(N, 3)``.
    opacity : torch.Tensor or np.ndarray, optional
        Per-Gaussian opacity weights, shape ``(N, 1)`` or ``(N,)``.
        If given the centroid is opacity-weighted.

    Returns
    -------
    np.ndarray
        Shape ``(3,)``.
    """
    if isinstance(gaussian_xyz, torch.Tensor):
        gaussian_xyz = gaussian_xyz.detach().cpu().numpy()
    if isinstance(opacity, torch.Tensor):
        opacity = opacity.detach().cpu().numpy()

    if opacity is not None:
        opacity = opacity.squeeze()
        return (gaussian_xyz * opacity[:, None]).sum(axis=0) / opacity.sum()
    return gaussian_xyz.mean(axis=0)


def estimate_gaussian_extent(gaussian_xyz, center):
    """Estimate bounding-sphere radius of Gaussians around *center*.

    Parameters
    ----------
    gaussian_xyz : torch.Tensor or np.ndarray
        Shape ``(N, 3)``.
    center : np.ndarray
        Shape ``(3,)``.

    Returns
    -------
    float
        Maximum distance from *center* to any Gaussian.
    """
    if isinstance(gaussian_xyz, torch.Tensor):
        gaussian_xyz = gaussian_xyz.detach().cpu().numpy()
    return float(np.linalg.norm(gaussian_xyz - center[None, :], axis=1).max())
