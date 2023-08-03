import os
import argparse
import trimesh
import pyrender
import imageio
import numpy as np
import matplotlib.pyplot as plt


def look_at(eye, center, up):
    eye = np.array(eye, dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    mat = np.eye(4)
    mat[0, :3] = right
    mat[1, :3] = down
    mat[2, :3] = -fwd
    mat[:3, 3] = -eye
    return mat


def render_mesh_video(obj_path, out_video, render_views, fps=12, transform=False):
    # Prepare image list
    images = []

    # load obj/glb file
    scene = trimesh.load(obj_path, force="scene", merge_primitives=True)
    print(f"Found {len(scene.geometry.values())} geometry")

    # Normalize the scene into a unit cube centered at the origin
    scene = scene.apply_transform(
        trimesh.transformations.scale_matrix(3.5 / scene.scale)
    )

    # Set the scene to be centered at the coordinate
    scene = scene.apply_transform(
        trimesh.transformations.translation_matrix(-scene.centroid)
    )

    # Transform the coordinates of the scene
    if transform:
        # xyz -> xzy
        scene = scene.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )

    # Define a rotation matrix to use for each rotation step
    rotation_step = trimesh.transformations.rotation_matrix(
        angle=(2 * np.pi / render_views), direction=[0, 0, 1]
    )

    # Create a Pyrender camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    # For each perspective, rotate the object, render the scene and save the image
    for _ in range(render_views):
        # Create a copy of the scene for rendering
        render_scene = pyrender.Scene.from_trimesh_scene(scene)

        # Add the camera to the scene
        eye = np.array([0, 4, 0])
        center = np.array([0, 0, 0])
        up = np.array([0, 0, -1])
        camera_pose = look_at(eye, center, up)
        render_scene.add(camera, pose=camera_pose)

        # Add full ambient lighting
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        render_scene.add(light, pose=camera_pose)

        # Create a Pyrender offscreen renderer
        renderer = pyrender.OffscreenRenderer(768, 768)

        # Render the scene
        color, _ = renderer.render(render_scene)

        # Add the image to the list
        images.append(color)

        # Save the image for debugging
        # plt.imsave("render_{}.png".format(i), color)
        # plt.imsave("render_depth_{}.png".format(i), depth)

        # Apply the rotation
        scene.apply_transform(rotation_step)

        renderer.delete()

    # Save the video
    imageio.mimsave(out_video, images, fps=fps)
    print(f"Saved to {out_video}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_in", type=str, help="path to mesh file")
    parser.add_argument("--pth_out", type=str, help="path to output render video")
    parser.add_argument("--render_views", type=int, default=36, help="number of views")
    parser.add_argument("--fps", type=int, default=12, help="fps of the video")
    parser.add_argument("--transform", action="store_true", help="xyz -> xzy")
    return parser.parse_args()


def main(args):
    if os.path.isdir(args.pth_in):
        os.makedirs(args.pth_out, exist_ok=True)
        for f in os.listdir(args.pth_in):
            if f.endswith(".obj") or f.endswith(".glb"):
                obj_path = os.path.join(args.pth_in, f)
                out_video = os.path.join(args.pth_out, f + ".mp4")
                print(f"Rendering {obj_path}")

                render_mesh_video(
                    obj_path=obj_path,
                    out_video=out_video,
                    render_views=args.render_views,
                    fps=args.fps,
                    transform=args.transform,
                )
    else:
        if "/" in args.pth_out:
            os.makedirs(os.path.dirname(args.pth_out), exist_ok=True)

        render_mesh_video(
            obj_path=args.pth_in,
            out_video=args.pth_out,
            render_views=args.render_views,
            fps=args.fps,
            transform=args.transform,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)

