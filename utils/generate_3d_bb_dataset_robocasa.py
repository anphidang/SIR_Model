"""
Offline extractor for 3D object bounding boxes.

Unlike the 2D pipeline (generate_graph_dataset_robocasa.py), the raw per-frame
segmentation/depth needed for this is not stored in the recorded demos -
only RGB images + the full simulator `states` are. So we replay each demo's
`states` through the simulator with segmentation + depth cameras enabled,
back-project each object's mask into a world-frame point cloud (fusing both
static camera views, since they land in the same coordinate frame), and fit
an oriented box: center(3) + extents(3) + 6D rotation (first two columns of
the rotation matrix) = 12 dims.

Originally a CloseSingleDoor-only proof of concept; now supports extracting
the full dataset for any task and persisting to a single HDF5 file
(bb3d_dataset.hdf5) - one group per demo, one (T, 12) array + (T,) validity
mask per object seen in that demo.

To use these boxes as a training modality, run this script first, then let
generate_3d_bb_graph_dataset_robocasa.py (invoked automatically by
RoboCasa_Manager via configs/manager/robocasa.yaml's graph_modalities:
["bb3d_coordinates"]) convert bb3d_dataset.hdf5 into the graph .pth files
the dataloader expects.
"""
import argparse
import json
import os

import h5py
import numpy as np
import robocasa  # noqa: F401 - registers RoboCasa kitchen tasks with robosuite
import robosuite
from tqdm import tqdm

from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

STATIC_VIEWS = ["robot0_agentview_left", "robot0_agentview_right"]
# Rendered only for this offline extraction (never touches the 128x128 RGB used
# for training) - higher res means denser point clouds for small objects like
# the gripper fingers.
RENDER_SIZE = 256
MIN_POINTS_PER_OBJECT = 20
# Depth values near the far clipping plane correspond to background/void
# (or mis-segmented silhouette edges) and blow up the box otherwise - a
# kitchen scene is a few meters across at most.
MAX_DEPTH_METERS = 4.0
# Robust min/max: guards against residual anti-aliased edge pixels that
# survive the depth clip.
BOX_PERCENTILE = 1.0


def build_env(demo_file: str, seed: int = 42):
    f = h5py.File(demo_file, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    config = env_meta["env_kwargs"]
    config["camera_heights"] = RENDER_SIZE
    config["camera_widths"] = RENDER_SIZE
    config["camera_depths"] = True
    # RoboCasa's Kitchen base class hardcodes camera_segmentations="class" itself
    # and doesn't accept it as a kwarg.
    config["seed"] = seed
    env = robosuite.make(**config, env_name=env_meta["env_name"])
    return env, f


def backproject_mask_to_world(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray):
    # Deliberately no per-view point-count gate here: the two static views are
    # merged before the real MIN_POINTS_PER_OBJECT threshold is applied, so an
    # object barely visible in one view (e.g. gripper fingers) still counts.
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None

    z = depth[ys, xs]
    valid = z < MAX_DEPTH_METERS
    if not np.any(valid):
        return None
    ys, xs, z = ys[valid], xs[valid], z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    cam_pts = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
    world_pts = (cam_to_world @ cam_pts.T).T[:, :3]
    return world_pts


def fit_oriented_box(points: np.ndarray):
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    local = centered @ axes
    mins = np.percentile(local, BOX_PERCENTILE, axis=0)
    maxs = np.percentile(local, 100 - BOX_PERCENTILE, axis=0)
    extents = maxs - mins
    center = mean + axes @ ((mins + maxs) / 2)
    rot6d = axes[:, :2].reshape(-1)
    return center, extents, rot6d


def save_episode_boxes(out_file: h5py.File, ep: str, per_frame: list):
    # per_frame[t] is a dict {obj_name: (12,) vec} for frame t - sparse because
    # not every object is visible/segmented in every frame. Reshaped here into
    # one fixed-length (T, 12) array + (T,) validity mask per object, which is
    # much easier to consume downstream than a list of per-frame dicts.
    if ep in out_file:
        del out_file[ep]
    grp = out_file.create_group(ep)
    num_frames = len(per_frame)
    obj_names = sorted({name for frame in per_frame for name in frame})
    for name in obj_names:
        boxes = np.full((num_frames, 12), np.nan, dtype=np.float32)
        valid = np.zeros((num_frames,), dtype=bool)
        for t, frame in enumerate(per_frame):
            if name in frame:
                boxes[t] = frame[name]
                valid[t] = True
        obj_grp = grp.create_group(name)
        obj_grp.create_dataset("boxes", data=boxes)
        obj_grp.create_dataset("valid", data=valid)
    grp.attrs["num_frames"] = num_frames
    out_file.flush()


def extract_task(
    dataset_path: str,
    task_name: str,
    num_demos: int = None,
    num_frames: int = None,
    seed: int = 42,
    output_path: str = None,
    resume: bool = True,
):
    demo_file = os.path.join(dataset_path, task_name, "demo_gentex_im128_randcams.hdf5")
    env, f = build_env(demo_file, seed=seed)

    demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    if num_demos is not None:
        demo_keys = demo_keys[:num_demos]

    out_file = None
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_file = h5py.File(output_path, "a" if resume else "w")
        if resume:
            already_done = set(out_file.keys())
            demo_keys = [ep for ep in demo_keys if ep not in already_done]

    results = {}
    for ep in tqdm(demo_keys, desc=f"{task_name} demos"):
        grp = f["data"][ep]
        model_xml = grp.attrs["model_file"]
        ep_meta = json.loads(grp.attrs["ep_meta"]) if "ep_meta" in grp.attrs else {}
        states = grp["states"][()]
        if num_frames is not None:
            states = states[:num_frames]

        # Kitchen._reset_internal() re-picks a random fixture layout on every plain
        # env.reset() call. set_ep_meta pins it back to the layout that was actually
        # recorded, so the fixture python objects (e.g. env.door_fxtr) match the
        # joints that exist in this episode's model_file.
        if hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        elif hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        # classes_to_ids depends on which fixtures/distractors this specific
        # episode's model actually contains (varies per episode), so the
        # id<->name mapping must be rebuilt per episode, not once globally.
        cls_list = {cls: i for i, cls in enumerate(env.model.classes_to_ids.keys())}
        id_to_cls = {i + 1: cls for cls, i in cls_list.items()}

        per_frame = []
        for state in tqdm(states, desc=ep, leave=False):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            if hasattr(env, "update_state"):
                env.update_state()
            obs = env._get_observations(force_update=True)

            world_pts_per_obj = {}
            for view in STATIC_VIEWS:
                seg = obs[f"{view}_segmentation_class"][:, :, 0]
                depth_norm = obs[f"{view}_depth"]
                depth = get_real_depth_map(env.sim, depth_norm)[:, :, 0]
                K = get_camera_intrinsic_matrix(env.sim, view, RENDER_SIZE, RENDER_SIZE)
                cam_to_world = get_camera_extrinsic_matrix(env.sim, view)

                for obj_id in np.unique(seg):
                    if obj_id == 0 or obj_id not in id_to_cls:
                        continue
                    name = id_to_cls[obj_id]
                    pts = backproject_mask_to_world(seg == obj_id, depth, K, cam_to_world)
                    if pts is None:
                        continue
                    world_pts_per_obj.setdefault(name, []).append(pts)

            frame_boxes = {}
            for name, chunks in world_pts_per_obj.items():
                pts = np.concatenate(chunks, axis=0)
                if pts.shape[0] < MIN_POINTS_PER_OBJECT:
                    continue
                center, extents, rot6d = fit_oriented_box(pts)
                frame_boxes[name] = np.concatenate([center, extents, rot6d]).astype(np.float32)

            per_frame.append(frame_boxes)

        if out_file is not None:
            save_episode_boxes(out_file, ep, per_frame)
        else:
            results[ep] = per_frame

    if out_file is not None:
        out_file.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--task", default="CloseSingleDoor")
    # None = no cap (full dataset / full episode length).
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to an HDF5 file to persist results to (one group per demo). "
        "If omitted, results are printed instead of saved.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="With --output, overwrite the file instead of skipping demos already present in it.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None and (args.num_demos is None or args.num_demos > 1):
        # Full/large runs default to persisting rather than dumping to stdout.
        output_path = os.path.join(args.dataset_path, args.task, "bb3d_dataset.hdf5")

    results = extract_task(
        args.dataset_path,
        args.task,
        args.num_demos,
        args.num_frames,
        output_path=output_path,
        resume=not args.no_resume,
    )

    if output_path is not None:
        print(f"Saved 3D bounding boxes to {output_path}")
    else:
        for ep, frames in results.items():
            print(f"=== {ep} ===")
            for t, boxes in enumerate(frames):
                print(f" frame {t}:")
                for name, vec in sorted(boxes.items()):
                    c, e = vec[:3], vec[3:6]
                    print(f"   {name:20s} center={c.round(3)} extents={e.round(3)}")
