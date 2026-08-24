"""
Offline extractor for 3D object bounding boxes.

Unlike the 2D pipeline (generate_graph_dataset_robocasa.py), the raw per-frame
segmentation/depth needed for this is not stored in the recorded demos -
only RGB images + the full simulator `states` are. So we replay each demo's
`states` through the simulator with segmentation + depth cameras enabled,
back-project each object's mask into a world-frame point cloud (fusing both
static camera views, since they land in the same coordinate frame), and fit
an oriented box. The box is then re-expressed relative to the gripper (same
world_pose_in_gripper convention robocasa's own kitchen.py observables use),
under one of two FRAME_MODES:

- "full": rotate+translate into the gripper frame (SE(3)). Removes the
  absolute world pose, but couples the whole scene to the wrist orientation -
  OSC action deltas are in the world/base frame, so the model has to invert
  that rotation implicitly, and a small wrist rotation reorganizes every
  node's features at once.
- "translation_only": subtract only the gripper's world position, keep
  world-aligned axes. Removes the same absolute position but stays
  axis-consistent with the action space. Untested alternative, not a
  replacement - compare both empirically before picking one.

This does NOT eliminate the absolute-pose shortcut the SIR paper's positional
bias refers to, it only makes it harder to use. Any node whose own world pose
is fixed/known relative to the base (PandaMobile, and effectively any static
fixture - Wall, Floor, Counter, ...) still encodes the gripper's absolute pose
in the scene invertibly once expressed in the gripper frame: its
gripper-relative pose *is* (an affine function of) the world_pose_in_gripper
transform itself. Dropping a node does not close this - as long as any other
node with a known/fixed world pose remains in the graph, the shortcut is
still reconstructible from it.

center(3) + extents(3) + 6D rotation (first two columns of the rotation
matrix, NOT a quaternion - deliberately, to avoid the q/-q double-cover
discontinuity a near-identity relative orientation would otherwise produce
in the time series) = 12 dims. Extents are frame-invariant (box side
lengths), so only center and rotation go through the transform.
PandaGripper's own box collapses to (approximately) the frame origin/
identity rotation once expressed relative to itself - by construction, not
a bug - so it carries zero positional information in this modality.
build_node_feature() appends robot0_gripper_qpos (finger joint positions,
genuinely new information, not implied by the transform, still varies over
time) to that node only, zero-padding every other node, so
BB3D_FEATURE_DIM/RELEVENT_NODES/topk-k stay unchanged across nodes. Note for
interpreting sub-graphs/explanations: those two zero-padded dims are also a
perfectly separating "is this the gripper" indicator, so a sparsifier
selecting PandaGripper in this modality alone isn't evidence it learned
anything about the scene.

Unlike bb_coordinates (2D pixel boxes, normalized to [0,1] by dividing by
image size - see generate_graph_dataset_robocasa.py/get_bb_pos), these raw
features mix wildly different scales: positions/extents in meters (~0.02-2),
unitless rotation entries (~[-1,1]), qpos in meters (~0-0.04) - the qpos
channel especially would be 1-2 orders of magnitude smaller than the rest of
the vector and effectively invisible after the GNN's first linear layer.
normalize_feature() below fixes this: box dims against compute_feature_stats()
(data-driven, scene-dependent - no physical bound exists), gripper_qpos
against GRIPPER_QPOS_RANGE (the gripper's known physical joint limits, NOT
fitted from data - confirmed empirically that a 2-task push/close sample only
exercises about half that range, so a fitted std would be too small and blow
up on any task that actually opens/closes the gripper).
save_feature_stats()/load_feature_stats() persist the box stats so they can
be computed ONCE (ideally on the training split) and reused identically for
offline graph building AND live rollout - the same train/inference
consistency requirement as the frame transform itself, now also checked
mechanically (not just documented) via stats_hash(): create_3d_bb_graphs_and_save()
writes a hash of the stats each task's graphs were built with, and kitchen.py
verifies it at rollout instead of trusting the file hasn't drifted.

fit_oriented_box()'s PCA axes are also canonicalized (see _AXIS_SIGN_REFERENCE
below) - confirmed empirically that without it, ~2% of frame-to-frame rot6d
transitions are sign-flip discontinuities (many kitchen fixtures have two
similarly-sized in-plane axes), not real motion; canonicalizing against a
fixed reference roughly halves that on the same sample, though a handful of
more genuinely symmetric fixtures (Fridge, CoffeeMachine in one test) still
flip sometimes - a single fixed reference can't fully resolve true
degeneracies.

Originally a CloseSingleDoor-only proof of concept; now supports extracting
the full dataset for any task and persisting to a single HDF5 file
(bb3d_dataset.hdf5) - one group per demo, one (T, BB3D_FEATURE_DIM) array +
(T,) validity mask per object seen in that demo.

To use these boxes as a training modality, run this script first, then let
generate_3d_bb_graph_dataset_robocasa.py (invoked automatically by
RoboCasa_Manager via configs/manager/robocasa.yaml's graph_modalities:
["bb3d_coordinates"]) convert bb3d_dataset.hdf5 into the graph .pth files
the dataloader expects.
"""
import argparse
import hashlib
import json
import os

import h5py
import numpy as np
import robocasa  # noqa: F401 - registers RoboCasa kitchen tasks with robosuite
import robosuite
import robosuite.utils.transform_utils as T
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

GRIPPER_NODE_NAME = "PandaGripper"
GRIPPER_QPOS_DIM = 2  # PandaGripper's two (mirrored) finger joint positions
# robosuite panda_gripper.xml: finger_joint1 range="0.0 0.04", finger_joint2 range="-0.04 0.0"
# (verified against the actual MJCF, not assumed). Fixed physical range, not a fitted
# mean/std - see normalize_feature()'s docstring for why.
GRIPPER_QPOS_RANGE = np.array([[0.0, 0.04], [-0.04, 0.0]], dtype=np.float32)
GRIPPER_QPOS_CENTER = GRIPPER_QPOS_RANGE.mean(axis=1)
GRIPPER_QPOS_HALF_RANGE = (GRIPPER_QPOS_RANGE[:, 1] - GRIPPER_QPOS_RANGE[:, 0]) / 2
BOX_FEATURE_DIM = 12  # center(3) + extents(3) + 6D rotation(6)
BB3D_FEATURE_DIM = BOX_FEATURE_DIM + GRIPPER_QPOS_DIM
FRAME_MODES = ("full", "translation_only")
# Canonicalizes the sign of fit_oriented_box()'s first two principal axes (the only ones
# that ever reach the exported rot6d feature - see world_box_to_gripper_frame()). PCA
# eigenvectors are only defined up to sign, and for objects with two similarly-sized
# in-plane axes (drawers, doors, boxes, walls - most of a kitchen) that sign flips
# frame-to-frame. Confirmed empirically on a small extracted sample: pooled |rot6d[t] -
# rot6d[t-1]| across all objects is bimodal - median ~0.008 (real motion) but max ~2.0 in
# every dimension with 1-3% of transitions >1.0, exactly the ~2x-axis-length jump a sign
# flip produces. A fixed world-frame reference doesn't fully resolve true degeneracies
# (two exactly-equal eigenvalues), only a consistent, deterministic 2-fold sign choice.


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


_AXIS_SIGN_REFERENCE = np.eye(3, dtype=np.float64)


def fit_oriented_box(points: np.ndarray):
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    # Canonicalize the sign of the two axes that reach the exported rot6d feature (see
    # FRAME_MODES comment above) against a fixed world-frame reference, so the same
    # geometry always yields the same sign instead of an arbitrary PCA-eigenvector choice.
    for i in range(2):
        if axes[:, i] @ _AXIS_SIGN_REFERENCE[:, i] < 0:
            axes[:, i] *= -1

    local = centered @ axes
    mins = np.percentile(local, BOX_PERCENTILE, axis=0)
    maxs = np.percentile(local, 100 - BOX_PERCENTILE, axis=0)
    extents = maxs - mins
    center = mean + axes @ ((mins + maxs) / 2)
    # world-frame center + full rotation matrix; reduced to gripper-relative
    # center + 6D rotation by world_box_to_gripper_frame() at the call site.
    return center, extents, axes


def get_world_pose_in_gripper(obs: dict, pf: str):
    """4x4 transform mapping world-frame poses into the gripper frame, mirroring
    robocasa's own world_pose_in_gripper sensor (kitchen.py)."""
    return T.pose_inv(T.pose2mat((obs[f"{pf}eef_pos"], obs[f"{pf}eef_quat"])))


def world_box_to_gripper_frame(center: np.ndarray, axes: np.ndarray, world_pose_in_gripper: np.ndarray):
    """FRAME_MODES="full": rotate+translate into the gripper frame (SE(3))."""
    obj_pose = np.eye(4)
    obj_pose[:3, :3] = axes
    obj_pose[:3, 3] = center
    rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, world_pose_in_gripper)
    rel_center = rel_pose[:3, 3]
    rel_rot6d = rel_pose[:3, :2].reshape(-1)
    return rel_center, rel_rot6d


def world_box_to_recentered_frame(center: np.ndarray, axes: np.ndarray, gripper_pos_world: np.ndarray):
    """FRAME_MODES="translation_only": subtract the gripper's world position only, keep
    world-aligned axes. Stays axis-consistent with world/base-frame OSC action deltas,
    unlike world_box_to_gripper_frame() which also rotates the whole scene by the wrist
    orientation. Untested alternative - compare against "full" empirically."""
    rel_center = center - gripper_pos_world
    rot6d = axes[:, :2].reshape(-1)
    return rel_center, rot6d


def transform_box_to_node_frame(
    center: np.ndarray,
    axes: np.ndarray,
    world_pose_in_gripper: np.ndarray,
    gripper_pos_world: np.ndarray,
    frame_mode: str,
):
    if frame_mode == "full":
        return world_box_to_gripper_frame(center, axes, world_pose_in_gripper)
    elif frame_mode == "translation_only":
        return world_box_to_recentered_frame(center, axes, gripper_pos_world)
    raise ValueError(f"Unknown frame_mode: {frame_mode!r}, expected one of {FRAME_MODES}")


def build_node_feature(name: str, rel_center: np.ndarray, extents: np.ndarray, rel_rot6d: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    """Box feature + gripper_qpos, zero-padded for every node except GRIPPER_NODE_NAME.

    PandaGripper's own box collapses to ~the frame origin/identity rotation once expressed
    relative to itself, so it carries no positional information here regardless of
    frame_mode. gripper_qpos is attached only to that node since it's genuinely new signal,
    not implied by the transform, and still varies over time.
    """
    box = np.concatenate([rel_center, extents, rel_rot6d]).astype(np.float32)
    extra = (
        np.asarray(gripper_qpos, dtype=np.float32)
        if name == GRIPPER_NODE_NAME
        else np.zeros(GRIPPER_QPOS_DIM, dtype=np.float32)
    )
    return np.concatenate([box, extra])


FEATURE_STATS_FILENAME = "bb3d_feature_stats.json"


def _collect_rows_by_name(dataset_path: str, task_names: list = None) -> dict:
    """Scans <dataset_path>/<task>/bb3d_dataset.hdf5 for every task in task_names (or every
    task under dataset_path that has one, if task_names is None) and returns
    {object_name: (N, BB3D_FEATURE_DIM) array} of every valid (object, frame) row pooled
    across all of them."""
    if task_names is None:
        task_names = sorted(
            name for name in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, name, "bb3d_dataset.hdf5"))
        )

    rows_by_name = {}
    for task_name in task_names:
        path = os.path.join(dataset_path, task_name, "bb3d_dataset.hdf5")
        with h5py.File(path, "r") as f:
            for ep in f.keys():
                grp = f[ep]
                for name in grp.keys():
                    boxes = grp[name]["boxes"][()]
                    valid = grp[name]["valid"][()]
                    rows = boxes[valid]
                    if rows.shape[0] == 0:
                        continue
                    rows_by_name.setdefault(name, []).append(rows)

    return {name: np.concatenate(chunks, axis=0) for name, chunks in rows_by_name.items()}


def compute_feature_stats(dataset_path: str, task_names: list = None, eps: float = 1e-6) -> dict:
    """Per-dimension mean/std of the box feature (dims 0:BOX_FEATURE_DIM), pooled over every
    valid (object, frame) row across the given tasks' bb3d_dataset.hdf5 (already generated
    via extract_task/this script's CLI) - position/extent/rotation scale is scene-dependent,
    not physically bounded, so this has to be fit from data.

    gripper_qpos (dims BOX_FEATURE_DIM:) is intentionally NOT included here - normalize_feature()
    scales it against GRIPPER_QPOS_RANGE, the gripper's known physical joint range, instead of
    a fitted mean/std. Fitting it from data was tried and measurably wrong: on a 2-task sample
    of push/close tasks the gripper aperture barely varies (raw range ~0.021-0.039 out of the
    physical 0.0-0.04), so the fitted std would be artificially tiny and blow up on any task
    that actually opens/closes the gripper (e.g. pick-and-place) - exactly the kind of
    train/eval scale mismatch this whole normalization step exists to prevent.

    Freeze this ONCE (ideally on the training split) via save_feature_stats() and reuse the
    same file everywhere via load_feature_stats() - recomputing it separately per split/task
    would reintroduce the same train/inference drift risk the frame transform itself had to
    be guarded against.
    """
    rows_by_name = _collect_rows_by_name(dataset_path, task_names)
    if not rows_by_name:
        raise ValueError(f"No bb3d_dataset.hdf5 found under {dataset_path} for tasks={task_names}")

    box_rows = np.concatenate([rows[:, :BOX_FEATURE_DIM] for rows in rows_by_name.values()], axis=0)
    return {
        "box_mean": box_rows.mean(axis=0).tolist(),
        "box_std": (box_rows.std(axis=0) + eps).tolist(),
    }


def save_feature_stats(stats: dict, dataset_path: str) -> str:
    out_path = os.path.join(dataset_path, FEATURE_STATS_FILENAME)
    with open(out_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    return out_path


def load_feature_stats(dataset_path: str) -> dict:
    path = os.path.join(dataset_path, FEATURE_STATS_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing. Generate bb3d_dataset.hdf5 for your training tasks first, "
            f"then run:\n  python utils/generate_3d_bb_dataset_robocasa.py --stats_only "
            f"--dataset_path {dataset_path}"
        )
    with open(path) as fh:
        return json.load(fh)


def stats_hash(stats: dict) -> str:
    """Short fingerprint of a feature-stats dict, so a consumer can assert it's using the
    exact stats file the graphs it's paired with were normalized with - "MUST be the same
    file" as a comment doesn't stop someone from regenerating bb3d_feature_stats.json on a
    different task split and silently shifting the scale rollout sees vs. what training saw.
    See generate_3d_bb_graph_dataset_robocasa.py (writer) and kitchen.py (checker)."""
    return hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()[:16]


def stats_hash_path(dataset_path: str, task_name: str, graph_modality: str) -> str:
    return os.path.join(dataset_path, task_name, f"{graph_modality}_stats_hash.txt")


def normalize_feature(name: str, vec: np.ndarray, stats: dict) -> np.ndarray:
    """Standardizes a build_node_feature() vector: box dims (0:BOX_FEATURE_DIM) against
    frozen data-driven stats from compute_feature_stats(); gripper_qpos dims against the
    gripper's fixed physical joint range (GRIPPER_QPOS_RANGE), not a fitted mean/std - see
    compute_feature_stats()'s docstring for why data-driven qpos stats don't generalize
    across tasks. Non-gripper qpos dims are left at exactly 0 (not shifted) - they're a
    placeholder, not a sample, and 0 stays the cleanest representation of "no
    gripper_qpos information for this node"."""
    box = (vec[:BOX_FEATURE_DIM] - np.asarray(stats["box_mean"], dtype=np.float32)) / np.asarray(stats["box_std"], dtype=np.float32)
    if name == GRIPPER_NODE_NAME:
        qpos = (vec[BOX_FEATURE_DIM:] - GRIPPER_QPOS_CENTER) / GRIPPER_QPOS_HALF_RANGE
    else:
        qpos = np.zeros(GRIPPER_QPOS_DIM, dtype=np.float32)
    return np.concatenate([box, qpos]).astype(np.float32)


def print_feature_diagnostics(dataset_path: str, task_names: list = None):
    """Prints per-dimension mean/std pooled across all nodes (to see the raw scale
    mismatch normalize_feature() is meant to fix), plus GRIPPER_NODE_NAME's and
    PandaMobile's own box std in isolation (near-zero for the former confirms it carries
    no positional signal pre-normalization; PandaMobile's shows how much of the absolute
    gripper pose is still reconstructible through it, see the module docstring)."""
    rows_by_name = _collect_rows_by_name(dataset_path, task_names)
    if not rows_by_name:
        print(f"No bb3d_dataset.hdf5 found under {dataset_path} for tasks={task_names}")
        return

    all_rows = np.concatenate(list(rows_by_name.values()), axis=0)
    dim_names = (
        [f"center_{a}" for a in "xyz"] + [f"extent_{a}" for a in "xyz"]
        + [f"rot6d_{i}" for i in range(6)] + [f"qpos_{i}" for i in range(GRIPPER_QPOS_DIM)]
    )
    print(f"=== per-dimension stats over {all_rows.shape[0]} rows, {len(rows_by_name)} node types ===")
    for dim, (dname, mean, std) in enumerate(zip(dim_names, all_rows.mean(axis=0), all_rows.std(axis=0))):
        print(f"   dim {dim:2d} {dname:10s} mean={mean:+.4f} std={std:.4f}")

    for name in (GRIPPER_NODE_NAME, "PandaMobile"):
        rows = rows_by_name.get(name)
        if rows is None:
            print(f"=== {name}: no rows found ===")
            continue
        box_std = rows[:, :BOX_FEATURE_DIM].std(axis=0)
        print(f"=== {name} ({rows.shape[0]} rows) box std ===")
        for dname, std in zip(dim_names[:BOX_FEATURE_DIM], box_std):
            print(f"   {dname:10s} std={std:.4f}")


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
        boxes = np.full((num_frames, BB3D_FEATURE_DIM), np.nan, dtype=np.float32)
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
    frame_mode: str = "full",
):
    if frame_mode not in FRAME_MODES:
        raise ValueError(f"Unknown frame_mode: {frame_mode!r}, expected one of {FRAME_MODES}")
    demo_file = os.path.join(dataset_path, task_name, "demo_gentex_im128_randcams.hdf5")
    env, f = build_env(demo_file, seed=seed)
    pf = env.robots[0].robot_model.naming_prefix

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
            world_pose_in_gripper = get_world_pose_in_gripper(obs, pf)
            gripper_pos_world = np.asarray(obs[f"{pf}eef_pos"], dtype=np.float32)
            gripper_qpos = np.asarray(obs[f"{pf}gripper_qpos"], dtype=np.float32)

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
                center, extents, axes = fit_oriented_box(pts)
                rel_center, rel_rot6d = transform_box_to_node_frame(
                    center, axes, world_pose_in_gripper, gripper_pos_world, frame_mode
                )
                frame_boxes[name] = build_node_feature(name, rel_center, extents, rel_rot6d, gripper_qpos)

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
    parser.add_argument(
        "--frame_mode",
        choices=FRAME_MODES,
        default="full",
        help="'full' rotates+translates into the gripper frame (SE(3)); 'translation_only' "
        "only subtracts the gripper's world position and keeps world-aligned axes, staying "
        "consistent with world/base-frame OSC action deltas. MUST match kitchen.py's "
        "RoboCasaKitchenTester.bb3d_frame_mode at inference time.",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Skip extraction. Print feature diagnostics and (re)compute+save "
        f"{FEATURE_STATS_FILENAME} over already-extracted tasks under --dataset_path, then exit.",
    )
    parser.add_argument(
        "--stats_tasks",
        nargs="*",
        default=None,
        help="With --stats_only: task subdirs to include (default: every task under "
        "--dataset_path that already has a bb3d_dataset.hdf5).",
    )
    args = parser.parse_args()

    if args.stats_only:
        print_feature_diagnostics(args.dataset_path, args.stats_tasks)
        stats = compute_feature_stats(args.dataset_path, args.stats_tasks)
        out_path = save_feature_stats(stats, args.dataset_path)
        print(f"Saved feature stats to {out_path}")
        raise SystemExit(0)

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
        frame_mode=args.frame_mode,
    )

    if output_path is not None:
        print(f"Saved 3D bounding boxes to {output_path}")
    else:
        for ep, frames in results.items():
            print(f"=== {ep} ===")
            for t, boxes in enumerate(frames):
                print(f" frame {t}:")
                for name, vec in sorted(boxes.items()):
                    c, e, qpos = vec[:3], vec[3:6], vec[12:]
                    print(f"   {name:20s} center({args.frame_mode})={c.round(3)} extents={e.round(3)} gripper_qpos={qpos.round(3)}")
