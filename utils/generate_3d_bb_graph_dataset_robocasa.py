"""
Converts the offline 3D bounding-box extraction (generate_3d_bb_dataset_robocasa.py,
bb3d_dataset.hdf5) into the same {modality}_left_image.pth / {modality}_right_image.pth
graph format that RoboCasaDataset (dataloader/datasets.py) already loads for every other
graph_modality (see generate_graph_dataset_robocasa.py for the 2D analog).

The 3D boxes are fused across both static cameras into a single world-frame estimate per
object/frame, so unlike the 2D modalities there is no genuine left/right split - the same
graph is written to both files. dataloader/utils.py's handle_fusing() takes the left copy
only when fusing views (mirroring how 'one_hot_labels' is handled), so the duplication is
just to satisfy RoboCasaDataset's file layout, not extra information.
"""
import os

import h5py
import networkx as nx
import torch
from tqdm import tqdm

from utils.create_graphs import create_graph_datapoint
from utils.generate_3d_bb_dataset_robocasa import (  # noqa: F401 - BB3D_FEATURE_DIM re-exported; single source of truth so this can't drift from the extractor
    BB3D_FEATURE_DIM,
    load_feature_stats,
    normalize_feature,
    stats_hash,
    stats_hash_path,
)


def create_3d_bb_graphs_and_save(
    dataset_path: str,
    task_name: str,
    graph_modality: str = "bb3d_coordinates",
):
    bb3d_path = os.path.join(dataset_path, task_name, "bb3d_dataset.hdf5")
    if not os.path.isfile(bb3d_path):
        raise Exception(
            f"{bb3d_path} is missing. Generate it first with:\n"
            f"  python utils/generate_3d_bb_dataset_robocasa.py --task {task_name} "
            f"--output {bb3d_path}"
        )
    # Frozen once (ideally on the training split) via --stats_only, reused identically here
    # and at rollout (envs/robocasa/kitchen.py) - raw box/qpos scales differ by orders of
    # magnitude (see generate_3d_bb_dataset_robocasa.py's module docstring), unnormalized
    # they'd be near-invisible to the GNN after the first linear layer.
    stats = load_feature_stats(dataset_path)
    with open(stats_hash_path(dataset_path, task_name, graph_modality), "w") as fh:
        fh.write(stats_hash(stats))

    f = h5py.File(bb3d_path, "r")
    demo_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))

    all_data = []
    for ep in tqdm(demo_keys, desc=f"{task_name} 3D-bb graphs"):
        grp = f[ep]
        num_frames = grp.attrs["num_frames"]
        obj_names = list(grp.keys())
        boxes = {name: grp[name]["boxes"][()] for name in obj_names}
        valid = {name: grp[name]["valid"][()] for name in obj_names}

        # Reused across all frames of this demo, like left_graph/right_graph in
        # generate_graph_dataset_robocasa.py: nodes persist once seen, objects missing
        # from a given frame just keep their last known feature instead of disappearing.
        graph = nx.DiGraph()
        demo_data = []
        for t in range(num_frames):
            frame_object_names = [name for name in obj_names if valid[name][t]]
            frame_objects = torch.stack(
                [torch.from_numpy(normalize_feature(name, boxes[name][t], stats)) for name in frame_object_names]
            ) if frame_object_names else torch.empty((0, BB3D_FEATURE_DIM))
            demo_data.append(create_graph_datapoint(graph, frame_object_names, frame_objects))
        all_data.append(demo_data)
    f.close()

    torch.save(all_data, os.path.join(dataset_path, task_name, graph_modality + "_left_image.pth"))
    torch.save(all_data, os.path.join(dataset_path, task_name, graph_modality + "_right_image.pth"))
