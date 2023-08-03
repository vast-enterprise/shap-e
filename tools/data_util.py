import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional, Union

import blobfile as bf
import numpy as np
import torch
from PIL import Image

from shap_e.rendering.blender.render import render_mesh, render_model
from shap_e.rendering.blender.view_data import BlenderViewData
from shap_e.rendering.mesh import TriMesh
from shap_e.rendering.point_cloud import PointCloud
from shap_e.rendering.view_data import ViewData
from shap_e.util.collections import AttrDict
from shap_e.util.image_util import center_crop, get_alpha, remove_alpha, resize


def load_or_create_pc(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    random_sample_count: int,
    point_count: int,
    num_views: int,
    verbose: bool = False,
) -> PointCloud:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if cache_dir is not None:
        cache_path = bf.join(
            cache_dir,
            f"pc_{bf.basename(path)}_mat_{num_views}_{random_sample_count}_{point_count}.npz",
        )
        if bf.exists(cache_path):
            return PointCloud.load(cache_path)
    else:
        cache_path = None

    with load_or_create_multiview(
        mesh_path=mesh_path,
        model_path=model_path,
        cache_dir=cache_dir,
        num_views=num_views,
        verbose=verbose,
    ) as mv:
        if verbose:
            print("extracting point cloud from multiview...")
        pc = mv_to_pc(
            multiview=mv, random_sample_count=random_sample_count, point_count=point_count
        )
        if cache_path is not None:
            pc.save(cache_path)
        return pc


@contextmanager
def load_or_create_multiview(
    *,
    mesh_path: Optional[str],
    model_path: Optional[str],
    cache_dir: Optional[str],
    num_views: int = 20,
    extract_material: bool = True,
    light_mode: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[BlenderViewData]:

    assert (model_path is not None) ^ (
        mesh_path is not None
    ), "must specify exactly one of model_path or mesh_path"
    path = model_path if model_path is not None else mesh_path

    if extract_material:
        assert light_mode is None, "light_mode is ignored when extract_material=True"
    else:
        assert light_mode is not None, "must specify light_mode when extract_material=False"

    if cache_dir is not None:
        if extract_material:
            cache_path = bf.join(cache_dir, f"mv_{bf.basename(path)}_mat_{num_views}.zip")
        else:
            cache_path = bf.join(cache_dir, f"mv_{bf.basename(path)}_{light_mode}_{num_views}.zip")
        if bf.exists(cache_path):
            with bf.BlobFile(cache_path, "rb") as f:
                yield BlenderViewData(f)
                return
    else:
        cache_path = None

    common_kwargs = dict(
        fast_mode=True,
        extract_material=extract_material,
        camera_pose="random",
        light_mode=light_mode or "uniform",
        verbose=verbose,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = bf.join(tmp_dir, "out.zip")
        if mesh_path is not None:
            mesh = TriMesh.load(mesh_path)
            render_mesh(
                mesh=mesh,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        elif model_path is not None:
            render_model(
                model_path,
                output_path=tmp_path,
                num_images=num_views,
                backend="BLENDER_EEVEE",
                **common_kwargs,
            )
        if cache_path is not None:
            bf.copy(tmp_path, cache_path)
        with bf.BlobFile(tmp_path, "rb") as f:
            yield BlenderViewData(f)


def mv_to_pc(multiview: ViewData, random_sample_count: int, point_count: int) -> PointCloud:
    pc = PointCloud.from_rgbd(multiview)

    # Handle empty samples.
    if len(pc.coords) == 0:
        pc = PointCloud(
            coords=np.zeros([1, 3]),
            channels=dict(zip("RGB", np.zeros([3, 1]))),
        )
    while len(pc.coords) < point_count:
        pc = pc.combine(pc)
        # Prevent duplicate points; some models may not like it.
        pc.coords += np.random.normal(size=pc.coords.shape) * 1e-4

    pc = pc.random_sample(random_sample_count)
    pc = pc.farthest_point_sample(point_count, average_neighbors=True)

    return pc
