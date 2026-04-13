"""End-to-end test for the Interactive Semantic Selector.

Usage
-----
    python test_pipeline.py --ply_path <path_to_ply>

The script will:
1. Load a Gaussian model from the given .ply file.
2. Generate orbital cameras around the Gaussian centroid.
3. Render preview images for each view.
4. Create mock masks (centre circle and half-plane) and extract
   Gaussian IDs for each.
5. Merge multi-view results.
6. (If a mesh path is provided) Transfer labels to mesh faces.
"""

import argparse
import os

import numpy as np
import torch

from semantic_selector.core import InteractiveSemanticSelector
from semantic_selector.camera_utils import (
    create_orbital_cameras,
    compute_gaussian_centroid,
    estimate_gaussian_extent,
)
from semantic_selector.sugar_loader import load_gaussian_model, load_gaussians_xyz


def _save_image(tensor, path):
    """Save a ``(C, H, W)`` float32 tensor to disk as a PNG."""
    import cv2
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = img[:, :, 0]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def make_circle_mask(h, w, cx, cy, radius):
    """Create a circular binary mask."""
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2).astype(np.float32)


def make_half_mask(h, w, axis="left"):
    """Create a half-plane binary mask."""
    mask = np.zeros((h, w), dtype=np.float32)
    if axis == "left":
        mask[:, : w // 2] = 1.0
    elif axis == "right":
        mask[:, w // 2 :] = 1.0
    elif axis == "top":
        mask[: h // 2, :] = 1.0
    else:
        mask[h // 2 :, :] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(description="Test Semantic Selector")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to .ply Gaussian model")
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="(Optional) Path to mesh for label transfer")
    parser.add_argument("--n_cameras", type=int, default=8,
                        help="Number of orbital views")
    parser.add_argument("--output_dir", type=str, default="test_output",
                        help="Directory for test outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------------
    print(f"[1/6] Loading model from {args.ply_path} ...")
    model = load_gaussian_model(args.ply_path)
    xyz_np = load_gaussians_xyz(model)
    print(f"       Loaded {xyz_np.shape[0]} Gaussians")

    # ---------------------------------------------------------------
    # 2. Create orbital cameras
    # ---------------------------------------------------------------
    center = compute_gaussian_centroid(xyz_np)
    radius = estimate_gaussian_extent(xyz_np, center) * 0.8
    print(f"[2/6] Centroid = {center}, orbital radius = {radius:.3f}")
    cameras = create_orbital_cameras(
        center=center, radius=radius, n_cameras=args.n_cameras,
    )

    # ---------------------------------------------------------------
    # 3. Render preview images
    # ---------------------------------------------------------------
    print(f"[3/6] Rendering {len(cameras)} preview views ...")
    selector = InteractiveSemanticSelector(model)
    renders = selector.render_views_for_selection(cameras)
    for i, r in enumerate(renders):
        path = os.path.join(args.output_dir, f"view_{i:03d}.png")
        _save_image(r["render"], path)
    print(f"       Saved previews to {args.output_dir}/view_*.png")

    # ---------------------------------------------------------------
    # 4. Mock mask → extract GIDs
    # ---------------------------------------------------------------
    print("[4/6] Extracting Gaussian IDs from mock masks ...")
    h, w = selector.image_height, selector.image_width
    all_gids = []

    # Test 1: all-white mask → should select all visible Gaussians
    white_mask = np.ones((h, w), dtype=np.float32)
    gids_white = selector.extract_gids_from_mask(cameras[0], white_mask)
    print(f"       All-white mask: {len(gids_white)} Gaussians selected")

    # Test 2: all-black mask → should return empty set
    black_mask = np.zeros((h, w), dtype=np.float32)
    gids_black = selector.extract_gids_from_mask(cameras[0], black_mask)
    print(f"       All-black mask: {len(gids_black)} Gaussians selected "
          f"(expected 0)")

    # Test 3: centre circle + half-plane masks from multiple views
    for i, cam in enumerate(cameras):
        # Alternate between circle and half-plane.
        if i % 2 == 0:
            mask = make_circle_mask(h, w, w // 2, h // 2, min(h, w) // 4)
        else:
            mask = make_half_mask(h, w, axis=["left", "right", "top", "bottom"][i % 4])
        gids = selector.extract_gids_from_mask(cam, mask)
        all_gids.append(gids)

    # ---------------------------------------------------------------
    # 5. Merge multi-view results
    # ---------------------------------------------------------------
    print("[5/6] Merging multi-view results ...")
    final_gids = selector.merge_multiview_gids(all_gids)
    print(f"       Union of {len(all_gids)} views: {len(final_gids)} Gaussians")
    for i, s in enumerate(all_gids):
        print(f"         View {i}: {len(s)} Gaussians")

    # ---------------------------------------------------------------
    # 6. (Optional) Mesh label transfer
    # ---------------------------------------------------------------
    if args.mesh_path and os.path.exists(args.mesh_path):
        print(f"[6/6] Transferring labels to mesh {args.mesh_path} ...")
        import trimesh
        from semantic_selector.mesh_transfer import transfer_labels_to_mesh, save_colored_mesh

        mesh = trimesh.load(args.mesh_path, force="mesh")
        mesh = transfer_labels_to_mesh(mesh, xyz_np, final_gids, "wall")
        out_path = os.path.join(args.output_dir, "labeled_mesh.ply")
        save_colored_mesh(mesh, out_path)
        print(f"       Saved labeled mesh to {out_path}")
    else:
        print("[6/6] Skipped mesh label transfer (no --mesh_path)")

    print("\nDone.")


if __name__ == "__main__":
    main()
