"""Transfer Gaussian labels to mesh faces via KNN."""

import numpy as np
from scipy.spatial import KDTree


# Default colour palette for semantic classes.
_DEFAULT_COLOURS = {
    "__selected__": np.array([220, 50, 50, 255], dtype=np.uint8),   # red
    "__unselected__": np.array([180, 180, 180, 255], dtype=np.uint8),  # grey
}


def transfer_labels_to_mesh(
    mesh,
    gaussians_xyz: np.ndarray,
    target_gids,
    class_name: str,
    k: int = 3,
    distance_threshold: float = 0.05,
):
    """Label mesh faces whose centroids are close to the selected Gaussians.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (will be modified in-place and returned).
    gaussians_xyz : np.ndarray
        All Gaussian positions, shape ``(N, 3)``.
    target_gids : set[int] or list[int]
        Indices of selected Gaussians.
    class_name : str
        Semantic label to assign (e.g. ``"wall"``, ``"door"``).
    k : int
        Number of nearest neighbours for KDTree query.
    distance_threshold : float
        Face-centroid distance threshold to consider a face "selected".

    Returns
    -------
    trimesh.Trimesh
        The same mesh object with face colours and metadata updated.
    """
    import trimesh

    target_gids = list(target_gids)
    if len(target_gids) == 0:
        face_labels = [None] * len(mesh.faces)
        mesh.metadata["face_labels"] = face_labels
        return mesh

    target_points = gaussians_xyz[target_gids]          # (M, 3)
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)  # (F, 3)

    tree = KDTree(target_points)
    distances, _ = tree.query(face_centroids, k=k)

    # For k=1 scipy returns a 1-D array; for k>1 a 2-D array.
    if distances.ndim == 1:
        min_dist = distances
    else:
        min_dist = distances.min(axis=1)

    selected_colour = _DEFAULT_COLOURS["__selected__"]
    unselected_colour = _DEFAULT_COLOURS["__unselected__"]

    face_labels = []
    colours = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    for i in range(len(mesh.faces)):
        if min_dist[i] < distance_threshold:
            face_labels.append(class_name)
            colours[i] = selected_colour
        else:
            face_labels.append(None)
            colours[i] = unselected_colour

    mesh.visual.face_colors = colours
    mesh.metadata["face_labels"] = face_labels
    return mesh


def save_colored_mesh(mesh, output_path: str):
    """Export a mesh (with face colours) to disk.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    output_path : str
        Destination path.  ``.ply`` and ``.obj`` are supported.
    """
    mesh.export(output_path)
