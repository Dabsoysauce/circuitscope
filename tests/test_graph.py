"""Unit tests for the computational graph (no model required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitscope.graph import ComputationalGraph, Node, NodeType


def test_edge_counts_and_connectivity():
    g = ComputationalGraph(n_layers=12, n_heads=12)
    assert len(g.sources) == 12 * 12 + 12 + 1            # heads + mlps + input
    assert len(g.dests) == 12 * 12 + 12 + 1              # heads + mlps + logits
    names = {(e.src.name, e.dst.name) for e in g.edges}
    # input feeds logits directly (residual skip)
    assert ("input", "logits") in names
    # nothing flows into input or out of logits
    assert not any(e.dst.name == "input" for e in g.edges)
    assert not any(e.src.name == "logits" for e in g.edges)


def test_intra_layer_ordering():
    g = ComputationalGraph(n_layers=4, n_heads=4)
    names = {(e.src.name, e.dst.name) for e in g.edges}
    # a head feeds the MLP in its own layer (attn precedes mlp)
    assert ("a0.h0", "mlp0") in names
    # heads in the same layer do not feed each other
    assert ("a2.h0", "a2.h1") not in names
    # no backward edges
    assert ("a3.h0", "a1.h0") not in names


def test_qkv_subedges():
    g = ComputationalGraph(n_layers=2, n_heads=2)
    qkv = {e.qkv for e in g.edges if e.dst.ntype == NodeType.HEAD}
    assert qkv == {"q", "k", "v"}
    # mlp/logits destinations carry no qkv tag
    assert all(e.qkv == "" for e in g.edges if e.dst.ntype != NodeType.HEAD)


def test_node_order_monotonic():
    n_layers = 6
    inp = Node(NodeType.INPUT)
    logits = Node(NodeType.LOGITS)
    assert inp.order(n_layers) < Node(NodeType.HEAD, 0, 0).order(n_layers)
    assert Node(NodeType.HEAD, 0, 0).order(n_layers) < Node(NodeType.MLP, 0).order(n_layers)
    assert Node(NodeType.MLP, 5).order(n_layers) < logits.order(n_layers)
