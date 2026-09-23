"""Systematic sub-graph inspection of a trained SIR checkpoint.

Replaces the one-episode debug print in Multi_XAI_GNN._log_selection_accuracy with an
aggregate over every frame of the validation split (and optionally a slice of the train
split): which nodes does the sparsifier retain, and how often is the actual target
(`door_obj`, or the fixture named in the instruction) among them?

Usage (from the repo root, `sir` conda env):
    python utils/inspect_subgraphs.py --ckpt logs/robocasa/2026-09-17/00-59-51 --out /tmp/x.json

The manager/method config is reloaded from `<ckpt>/.hydra/config.yaml`, exactly as
Trainer.check_if_model_is_loaded does for evaluation, except that the dataset is loaded so the
validation dataloader exists. Evaluation is deterministic top-k, so this is a property of the
checkpoint on demonstration frames - not of policy rollouts.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.generate_graph_dataset_robocasa import is_node_relevant_for_task  # noqa: E402


def build(ckpt, data_path, device):
    cfg = OmegaConf.load(os.path.join(ckpt, ".hydra", "config.yaml"))
    manager_cfg = cfg.manager
    manager_cfg.load_dataset = True
    manager_cfg.data_path = data_path
    method_cfg = cfg.method
    method_cfg.device = device

    manager = hydra.utils.instantiate(manager_cfg)

    # mirrors Trainer.set_runtime_parameter
    method_cfg.action_generator.proprio_dim = manager.prop_dim
    method_cfg.action_generator.graph_mod = manager.adapted_graph_modalities
    method_cfg.graph_encoder.input_dim = manager.graph_dim
    method_cfg.graph_encoder.edge_dim = manager.graph_edge_dim
    method_cfg.graph_encoder.modalities = manager.adapted_graph_modalities
    method = hydra.utils.instantiate(method_cfg)

    if manager.use_proprioceptive:
        method.instantiate_proprioceptive()
    if manager.use_image:
        method.instantiate_vision_encoder()
    if manager.use_graph:
        method.instantiate_graph_encoder()
    method.instantiate_optimizer()
    method.load_models(ckpt, "_final")
    method.eval()
    return manager, method, cfg


def attach_recorder(graph_encoder, sink):
    """`sink` is a one-element list holding the list that currently receives records."""
    orig = graph_encoder._log_selection_accuracy

    def wrapped(graph, sampled_nodes, task_names, lang_goals=None):
        names = getattr(graph, "node_names", None)
        if names is not None:
            flat = [n for sub in names for n in sub] if names and isinstance(names[0], list) else list(names)
            if len(flat) == graph.x.shape[0]:
                batch = graph.batch.tolist()
                per_all, per_sel = defaultdict(list), defaultdict(list)
                for i, n in enumerate(flat):
                    per_all[batch[i]].append(n)
                for p in sampled_nodes.tolist():
                    per_sel[batch[p]].append(flat[p])
                for b in sorted(per_all):
                    goal = lang_goals[b] if lang_goals is not None and b < len(lang_goals) else ""
                    task = task_names[b] if task_names else "CloseSingleDoor"
                    sink[0].append({"all": per_all[b], "sel": per_sel[b], "goal": goal, "task": task})
        return orig(graph, sampled_nodes, task_names, lang_goals)

    graph_encoder._log_selection_accuracy = wrapped


def target_nodes(r):
    """Nodes counting as 'the target' for this frame: the door panel, or any node whose name
    contains the fixture word from the instruction ("close the microwave door" -> "microwave")."""
    words = (r["goal"] or "").split(" ")
    fixture = words[-2].lower() if len(words) >= 2 else None
    return [n for n in r["all"] if n == "door_obj" or (fixture and fixture in n.lower())]


def chance_hit(n_all, n_target, k):
    """P(at least one target node in a uniformly random k-subset of n_all nodes)."""
    if n_target == 0:
        return 0.0
    k = min(k, n_all)
    if n_all - n_target < k:
        return 1.0
    return 1.0 - math.comb(n_all - n_target, k) / math.comb(n_all, k)


def summarize(records):
    n = len(records)
    sel_counter, avail_counter = Counter(), Counter()
    door_obj_sel = door_obj_avail = target_any = fixture_target_sel = 0
    gripper_sel = mobile_sel = 0
    chance_target_any = chance_door_obj = 0.0
    sel_sizes, all_sizes = [], []
    for r in records:
        all_sizes.append(len(r["all"]))
        sel_sizes.append(len(r["sel"]))
        for nme in r["all"]:
            avail_counter[nme] += 1
        for nme in r["sel"]:
            sel_counter[nme] += 1
        if "door_obj" in r["all"]:
            door_obj_avail += 1
            if "door_obj" in r["sel"]:
                door_obj_sel += 1
        tset = set(target_nodes(r))
        has_fixture_target = any(nme in tset and nme != "door_obj" for nme in r["sel"])
        if has_fixture_target:
            fixture_target_sel += 1
        if any(nme in tset for nme in r["sel"]):
            target_any += 1
        chance_target_any += chance_hit(len(r["all"]), len(tset), len(r["sel"]))
        if "door_obj" in r["all"]:
            chance_door_obj += chance_hit(len(r["all"]), 1, len(r["sel"]))
        if "PandaGripper" in r["sel"]:
            gripper_sel += 1
        if "PandaMobile" in r["sel"]:
            mobile_sel += 1
    return {
        "n_frames": n,
        "mean_nodes_available": sum(all_sizes) / max(n, 1),
        "mean_nodes_selected": sum(sel_sizes) / max(n, 1),
        "frac_door_obj_selected_when_available": door_obj_sel / max(door_obj_avail, 1),
        "chance_door_obj_selected": chance_door_obj / max(door_obj_avail, 1),
        "chance_target_selected_any": chance_target_any / max(n, 1),
        "frac_door_obj_available": door_obj_avail / max(n, 1),
        "frac_fixture_named_in_instruction_selected": fixture_target_sel / max(n, 1),
        "frac_target_selected_any": target_any / max(n, 1),
        "frac_PandaGripper_selected": gripper_sel / max(n, 1),
        "frac_PandaMobile_selected": mobile_sel / max(n, 1),
        "selection_frequency": {
            k: {"selected": v, "available": avail_counter[k], "rate": v / avail_counter[k]}
            for k, v in sorted(sel_counter.items(), key=lambda kv: -kv[1] / avail_counter[kv[0]])
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_path", default="/home/an-phi/Dokumente/rob/SIR_Model/data/")
    ap.add_argument("--train_batches", type=int, default=0, help="also run this many train batches")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val_loss", action="store_true",
                    help="also report mean validation loss, to check the loaded data/cache matches "
                         "the conditions the checkpoint was trained under (compare to its main.log)")
    args = ap.parse_args()

    manager, method, cfg = build(args.ckpt, args.data_path, args.device)
    records_val, records_train = [], []
    enc = method.graph_encoder
    sink = [records_val]
    attach_recorder(enc, sink)

    val_losses = []
    with torch.no_grad():
        for batch in manager.valid_dataloader:
            if args.val_loss:
                state, action, goal = method.preprocess_batch(batch)
                val_losses.append(method.compute_validation_loss(state, action['action'], goal)['total_loss'].item())
            else:
                method.preprocess_batch(batch)
    if val_losses:
        print(f"MEAN_VAL_TOTAL_LOSS {sum(val_losses) / len(val_losses):.4f} over {len(val_losses)} batches")

    if args.train_batches:
        sink[0] = records_train
        with torch.no_grad():
            for i, batch in enumerate(manager.train_dataloader):
                if args.train_batches > 0 and i >= args.train_batches:
                    break
                method.preprocess_batch(batch)

    out = {
        "ckpt": args.ckpt,
        "graph_modalities": list(cfg.manager.graph_modalities),
        "mask_objects": list(cfg.manager.get("mask_objects", []) or []),
        "sampling_strategy": cfg.method.graph_encoder.sparsification_layer.sampling_strategy,
        "val": summarize(records_val),
    }
    if val_losses:
        out["val_total_loss"] = sum(val_losses) / len(val_losses)
    if records_train:
        out["train_subset"] = summarize(records_train)
        out["all_frames"] = summarize(records_val + records_train)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    v = out.get("all_frames", out["val"])
    print(json.dumps({k: v[k] for k in v if k != "selection_frequency"}, indent=2))
    print("top selection rates:")
    for k, d in list(v["selection_frequency"].items())[:10]:
        print(f"  {k:24s} {d['selected']:5d}/{d['available']:5d} = {d['rate']:.3f}")


if __name__ == "__main__":
    main()
