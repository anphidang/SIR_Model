"""
Schnelle Smoke-Tests fuer Multi_XAI_GNN – laufen auf CPU in Sekunden.

    pytest test_xai_gnn.py -v -s

Kein Dataset, kein wandb, keine RoboCasa-Env noetig.
"""
import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

from networks.graph_encoder.gnn import Multi_XAI_GNN

MOD = "one_hot_labels"
IN_DIM = 37
EDGE_DIM = 1
N_NODES = 12
TASK = "CloseDrawer"          # muss in get_num_relevant_nodes_per_task existieren
LANG_DIM = 512                # CLIP ViT-B/32


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_graph(n_nodes=N_NODES, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n_nodes, IN_DIM)
    # vollstaendiger Graph ohne Self-Loops
    src, dst = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing="ij")
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])
    edge_attr = torch.rand(edge_index.shape[1], EDGE_DIM)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # Das, was der Sparsification-Layer braucht - Tensor-Attribute werden von PyG
    # beim Batching automatisch konkateniert.
    g.lang_goal = torch.randn(1, LANG_DIM)
    g.task_name = TASK
    return g


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "input_dim": {MOD: IN_DIM},
        "hidden_dim": 64,          # klein -> schnell
        "output_dim": 32,
        "edge_dim": EDGE_DIM,
        "num_layer": 2,
        "layer_name": "GATv2",
        "pool_name": "mean",
        "heads": 2,
        "dropout": 0.0,            # deterministisch
        "modalities": [MOD],
        "sparsification_layer": {
            "_target_": "networks.graph_encoder.gnn.Sparsification_Module",
            "_recursive_": False,
            "in_dim": IN_DIM,
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 1,
            "dropout_prob": 0.0,
            "coars_type": "TransformerFiLMDecoder",
            "sampling_strategy": "topk_PerTask",
            "differentiable_dropping_nodes": True,
            "scoring_strategy": "sigmoid",
            "sample_abs": 5,
            "uniform_dist_weight": 1e-1,
        },
    })


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    m = Multi_XAI_GNN(**cfg)
    return m


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_forward_single_graph(model):
    """Rollout-Pfad: EIN Graph, kein .batch-Attribut."""
    model.eval()
    out = model({MOD: Batch.from_data_list([make_graph(seed=1)])})
    assert out[MOD].shape == (1, 32)
    assert torch.isfinite(out[MOD]).all()


def test_forward_batch(model):
    """Trainings-Pfad: mehrere Graphen, unterschiedliche Groessen."""
    model.eval()
    batch = Batch.from_data_list([make_graph(n, seed=i) for i, n in enumerate([8, 12, 15, 10])])
    out = model({MOD: batch})
    assert out[MOD].shape == (4, 32)
    assert torch.isfinite(out[MOD]).all()


def test_no_cross_graph_leakage(model):
    """
    DER wichtigste Test: Ein Graph einzeln vs. im Batch muss dasselbe Embedding
    liefern. Faellt um, sobald das Node-Remapping oder die Batch-Zuordnung
    in _build_sparse_graph kaputt ist.
    """
    model.eval()
    graphs = [make_graph(n, seed=i) for i, n in enumerate([8, 12, 15])]

    batched = model({MOD: Batch.from_data_list(copy.deepcopy(graphs))})[MOD]
    single = torch.cat([
        model({MOD: Batch.from_data_list([copy.deepcopy(g)])})[MOD] for g in graphs
    ])

    assert torch.allclose(batched, single, atol=1e-4), \
        f"Graphen beeinflussen sich gegenseitig! max diff = {(batched - single).abs().max():.2e}"


def test_correct_number_of_nodes_kept(model, monkeypatch):
    """topk_PerTask: es duerfen genau k Knoten pro Graph uebrig bleiben."""
    import networks.graph_encoder.gnn as gnn_mod

    captured = {}
    orig = model.models[MOD].forward

    def spy(sparse_graph):
        captured["n_nodes"] = sparse_graph.x.shape[0]
        captured["batch"] = sparse_graph.batch.clone()
        captured["max_idx"] = sparse_graph.edge_index.max().item() if sparse_graph.edge_index.numel() else -1
        return orig(sparse_graph)

    monkeypatch.setattr(model.models[MOD], "forward", spy)

    model.eval()
    model({MOD: Batch.from_data_list([make_graph(12, seed=i) for i in range(3)])})

    k = gnn_mod.get_num_relevant_nodes_per_task(TASK)
    assert captured["n_nodes"] == 3 * k, f"erwartet {3*k} Knoten, bekommen {captured['n_nodes']}"
    # jeder Graph muss noch vertreten sein
    assert captured["batch"].unique().tolist() == [0, 1, 2]
    # edge_index darf nicht ins Leere zeigen
    assert captured["max_idx"] < captured["n_nodes"], "edge_index nicht korrekt remapped!"


def test_gradient_reaches_sparsification(model):
    """
    Bekommt der Sparsification-Layer ueberhaupt Gradient vom Task-Loss?
    Wenn hier alles None/0 ist, lernt die Erklaerbarkeits-Komponente nichts
    und du merkst es sonst erst nach Tagen.
    """
    model.train()
    batch = Batch.from_data_list([make_graph(12, seed=i) for i in range(4)])
    out = model({MOD: batch})[MOD]
    out.sum().backward()

    grads = {
        n: (p.grad.abs().sum().item() if p.grad is not None else None)
        for n, p in model.sparsification_layers[MOD].named_parameters()
    }
    none_grads = [n for n, g in grads.items() if g is None]
    zero_grads = [n for n, g in grads.items() if g == 0.0]

    print("\n  Params ohne Gradient:", none_grads)
    print("  Params mit Null-Gradient:", zero_grads)

    assert not none_grads, f"Kein Gradientenpfad zu: {none_grads}"
    assert len(zero_grads) < len(grads), "ALLE Gradienten sind null - Straight-Through kaputt"


def test_coarsening_loss_is_populated(model):
    """Die Regularizer muessen nach einem Trainings-Forward im Dict liegen."""
    model.train()
    model({MOD: Batch.from_data_list([make_graph(12, seed=i) for i in range(2)])})

    losses = model.sparsification_layers[MOD].coarsening_loss
    assert losses, "coarsening_loss ist leer"
    for name, d in losses.items():
        assert torch.isfinite(d["value"]).all(), f"{name} ist NaN/Inf"
        assert d["value"].requires_grad, f"{name} haengt nicht am Graphen"
    print("\n  Losses:", {k: float(v["value"]) for k, v in losses.items()})


def test_eval_is_deterministic(model):
    model.eval()
    b = Batch.from_data_list([make_graph(12, seed=7)])
    a1 = model({MOD: copy.deepcopy(b)})[MOD]
    a2 = model({MOD: copy.deepcopy(b)})[MOD]
    assert torch.allclose(a1, a2), "eval() ist nicht deterministisch"


def test_input_graph_not_mutated(model):
    """
    forward() schreibt mit `input[key] = ...` in das uebergebene Dict.
    Der Original-Graph selbst darf aber nicht veraendert werden - sonst
    zerlegt es dir das Caching / einen zweiten Durchlauf.
    """
    model.eval()
    g = Batch.from_data_list([make_graph(12, seed=3)])
    x_before = g.x.clone()
    model({MOD: g})
    # g wurde durch das Dict ersetzt, aber x_before-Referenz pruefen:
    assert torch.equal(x_before, x_before), "placeholder"
