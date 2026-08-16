"""
Schnelle Smoke-Tests fuer Multi_XAI_GNN – laufen auf CPU in Sekunden.

    pytest test_xai_gnn.py -v -s

Kein Dataset, kein wandb, keine RoboCasa-Env noetig.
"""
import copy

import pytest
import torch
from torch_geometric.data import Data, Batch

from networks.graph_encoder.gnn import Multi_XAI_GNN
from utils.generate_graph_dataset_robocasa import get_num_relevant_nodes_per_task

MOD = "bb_coordinates_cropped_image_feature"   # gemergter Key (use_graph_fusion=True)
IN_DIM = 41          # bb_coordinates(10) + cropped_image_feature(31)
EDGE_DIM = 1
TASK = "TurnOffSinkFaucet"
LANG_DIM = 512       # CLIP ViT-B/32 -> muss == sparsification hidden_dim sein (FiLM)
K = get_num_relevant_nodes_per_task(TASK)      # erwartete Anzahl behaltener Knoten


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def make_graph(n_nodes=12, seed=None):
    """Synthetischer Graph im Format der echten Daten: Data(x, edge_index, edge_attr)."""
    g_ = torch.Generator().manual_seed(seed if seed is not None else 0)
    x = torch.randn(n_nodes, IN_DIM, generator=g_)
    src, dst = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing="ij")
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])
    edge_attr = torch.rand(edge_index.shape[1], generator=g_)   # 1-D, wie in den echten Daten
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def run(model, graphs, lang=None):
    """Ruft den Encoder so auf, wie diffusion.preprocess_batch es tut."""
    batch = Batch.from_data_list([copy.deepcopy(g) for g in graphs])
    n = len(graphs)
    if lang is None:
        lang = torch.randn(n, LANG_DIM, generator=torch.Generator().manual_seed(99))
    return model({MOD: batch}, lang_emb=lang, task_names=[TASK] * n)[MOD]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Multi_XAI_GNN(
        input_dim={MOD: IN_DIM},
        hidden_dim=64,
        output_dim=32,
        edge_dim=EDGE_DIM,
        num_layer=2,
        layer_name="GATv2",
        pool_name="mean",
        heads=2,
        dropout=0.0,                 # deterministisch
        modalities=[MOD],
        sparsification_layer={
            "_target_": "networks.graph_encoder.gnn.Sparsification_Module",
            "_recursive_": False,
            "in_dim": IN_DIM,
            "hidden_dim": LANG_DIM,  # FiLM: film_cond_dim == embed_dim == 512
            "num_heads": 4,
            "num_layers": 1,
            "dropout_prob": 0.0,
            "coars_type": "TransformerFiLMDecoder",
            "sampling_strategy": "topk_PerTask",
            "differentiable_dropping_nodes": True,
            "scoring_strategy": "sigmoid",
            "sample_abs": 5,
            "uniform_dist_weight": 1e-1,
        },
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_forward_single_graph(model):
    """Rollout-Pfad: EIN Graph."""
    model.eval()
    out = run(model, [make_graph(12, seed=1)])
    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()


def test_forward_batch(model):
    """Trainings-Pfad: mehrere Graphen unterschiedlicher Groesse."""
    model.eval()
    out = run(model, [make_graph(n, seed=i) for i, n in enumerate([8, 12, 15, 10])])
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_selection_is_batch_independent(model, monkeypatch):
    """Die gewaehlten Knoten pro Graph duerfen nicht davon abhaengen,
    wer sonst im Batch liegt. (Embeddings tun das wegen PyG-LayerNorm
    im graph-mode sehr wohl - das ist upstream-Verhalten.)"""
    captured = []
    layer = model.sparsification_layers[MOD]
    orig = layer.forward

    def spy(*a, **kw):
        ew, nodes, probs = orig(*a, **kw)
        captured.append(nodes.clone())
        return ew, nodes, probs

    monkeypatch.setattr(layer, "forward", spy)
    model.eval()
    lang = torch.zeros(1, LANG_DIM)
    graphs = [make_graph(n, seed=i) for i, n in enumerate([8, 12, 15])]

    run(model, graphs, lang=lang.repeat(3, 1))
    batched_sel = captured[0]

    captured.clear()
    offsets, single_sel = 0, []
    for g in graphs:
        run(model, [g], lang=lang)
        single_sel.append(captured[-1] + offsets)
        offsets += g.x.shape[0]
    single_sel = torch.cat(single_sel)

    assert torch.equal(torch.sort(batched_sel).values, torch.sort(single_sel).values), \
        f"Selektion haengt von der Batch-Zusammensetzung ab!\n batch: {batched_sel}\n single: {single_sel}"


def test_correct_number_of_nodes_kept(model, monkeypatch):
    """topk_PerTask: genau k Knoten pro Graph, edge_index korrekt remapped."""
    captured = {}
    inner = model.models[MOD]
    orig = inner.forward

    def spy(sparse_graph):
        captured["n_nodes"] = sparse_graph.x.shape[0]
        captured["batch"] = sparse_graph.batch.clone()
        captured["max_idx"] = (sparse_graph.edge_index.max().item()
                               if sparse_graph.edge_index.numel() else -1)
        return orig(sparse_graph)

    monkeypatch.setattr(inner, "forward", spy)

    model.eval()
    run(model, [make_graph(12, seed=i) for i in range(3)])

    print(f"\n  k (aus RELEVENT_NODES[{TASK}]) = {K}")
    assert captured["n_nodes"] == 3 * K, \
        f"erwartet {3*K} Knoten, bekommen {captured['n_nodes']}"
    assert captured["batch"].unique().tolist() == [0, 1, 2], \
        "ein Graph hat alle Knoten verloren"
    assert captured["max_idx"] < captured["n_nodes"], \
        "edge_index zeigt ins Leere - Remapping kaputt!"


def test_gradient_reaches_sparsification(model):
    """Bekommt der Sparsification-Layer Gradient vom Task-Loss?"""
    model.train()
    out = run(model, [make_graph(12, seed=i) for i in range(4)])
    out.sum().backward()

    grads = {
        n: (p.grad.abs().sum().item() if p.grad is not None else None)
        for n, p in model.sparsification_layers[MOD].named_parameters()
    }
    none_grads = [n for n, g in grads.items() if g is None]
    zero_grads = [n for n, g in grads.items() if g == 0.0]


    dead = [n for n in none_grads if "cross_att" not in n]

    print(f"\n  ohne Gradient (gesamt): {len(none_grads)}/{len(grads)}")
    print(f"  Null-Gradient         : {len(zero_grads)}/{len(grads)}")
    print(f"  Kritisch ohne Pfad    : {len(dead)}")

    
    assert not dead, f"Kein Gradientenpfad zu: {dead[:5]}"
    assert len(zero_grads) < len(grads), "ALLE Gradienten null - Straight-Through kaputt"



def test_coarsening_loss_is_populated(model):
    """Regularizer muessen nach einem Trainings-Forward im Dict liegen."""
    model.train()
    run(model, [make_graph(12, seed=i) for i in range(2)])

    losses = model.sparsification_layers[MOD].coarsening_loss
    assert losses, "coarsening_loss ist leer"
    for name, d in losses.items():
        assert torch.isfinite(d["value"]).all(), f"{name} ist NaN/Inf"
        assert d["value"].requires_grad, f"{name} haengt nicht am Graphen"
    print("\n  Losses:", {k: round(float(v["value"]), 5) for k, v in losses.items()})


def test_eval_is_deterministic(model):
    model.eval()
    g = [make_graph(12, seed=7)]
    lang = torch.zeros(1, LANG_DIM)
    assert torch.allclose(run(model, g, lang=lang), run(model, g, lang=lang)), \
        "eval() ist nicht deterministisch"

# ans Ende von test_xai_gnn.py
from networks.graph_encoder.gnn import Multi_GNN

def test_baseline_gnn_has_same_leakage():
    torch.manual_seed(0)
    m = Multi_GNN(input_dim={MOD: IN_DIM}, hidden_dim=64, output_dim=32,
                  edge_dim=EDGE_DIM, num_layer=2, layer_name="GATv2",
                  pool_name="mean", heads=2, dropout=0.0, modalities=[MOD])
    m.eval()
    graphs = [make_graph(n, seed=i) for i, n in enumerate([8, 12, 15])]
    batched = m({MOD: Batch.from_data_list([copy.deepcopy(g) for g in graphs])})[MOD]
    single = torch.cat([m({MOD: Batch.from_data_list([copy.deepcopy(g)])})[MOD] for g in graphs])
    print(f"\n  Baseline-Multi_GNN max diff: {(batched-single).abs().max():.2e}")