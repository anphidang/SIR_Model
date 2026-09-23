"""Build the random_labels graph cache from the existing clip_labels cache.

Both label modalities have identical graph structure (one node per visible object, node_names
identical, same edges) and differ only in the per-node feature vector, which is a pure function
of the node's class name. So instead of re-running the (from-scratch broken, see
create_graphs_and_save) generator, load clip_labels_{left,right}_image.pth and swap each node's
feature for get_random_label_embeddings()[name].

Usage: python utils/make_random_label_cache.py --task CloseSingleDoor
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.generate_graph_dataset_robocasa import get_random_label_embeddings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--data_path", default="/home/an-phi/Dokumente/rob/SIR_Model/data/")
    args = ap.parse_args()

    emb = get_random_label_embeddings()
    for view in ("left", "right"):
        src = os.path.join(args.data_path, args.task, f"clip_labels_{view}_image.pth")
        dst = os.path.join(args.data_path, args.task, f"random_labels_{view}_image.pth")
        demos = torch.load(src, weights_only=False)
        n = 0
        for demo in demos:
            for g in demo:
                g.x = torch.stack([emb[name] for name in g.node_names]) if g.node_names else g.x
                n += 1
        torch.save(demos, dst)
        print(f"{view}: {len(demos)} demos, {n} graphs -> {dst}")


if __name__ == "__main__":
    main()
