"""The computational graph of a transformer, as nodes and residual-stream edges.

We model a HookedTransformer as a directed acyclic graph over *components*:

* ``input``            -- the embedding (token + positional), one source node.
* ``head L.H``         -- the output of attention head ``H`` in layer ``L``.
* ``mlp L``            -- the output of the MLP in layer ``L``.
* ``logits``           -- the final unembedding, one sink node.

Every component writes to and/or reads from the residual stream. Because the
residual stream is *additive*, an edge ``u -> d`` means "the output that ``u``
writes is read by ``d``". An edge is valid whenever ``u`` is computed strictly
before ``d``. We order components by ``layer * 2 + sublayer`` where attention is
sublayer 0 and the MLP is sublayer 1; ``input`` is ``-1`` (before everything)
and ``logits`` is ``2 * n_layers`` (after everything). This reproduces the exact
connectivity used by edge attribution patching (Syed et al., 2023).

Each *downstream* read happens through a distinct residual-shaped hook so the
edge can be scored independently:

* attention heads read via ``hook_q_input`` / ``hook_k_input`` / ``hook_v_input``
  (requires ``use_split_qkv_input=True``), so a single head is the destination
  of three logical sub-edges (q, k, v) per source;
* MLPs read via ``hook_mlp_in`` (requires ``use_hook_mlp_in=True``);
* ``logits`` reads the final ``hook_resid_post``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class NodeType(str, Enum):
    INPUT = "input"
    HEAD = "head"
    MLP = "mlp"
    LOGITS = "logits"


# Attention is computed before the MLP within a layer.
SUBLAYER_ATTN = 0
SUBLAYER_MLP = 1


@dataclass(frozen=True)
class Node:
    """A component in the computational graph."""

    ntype: NodeType
    layer: int = -1   # -1 for input/logits
    head: int = -1    # -1 unless ntype == HEAD

    @property
    def name(self) -> str:
        if self.ntype == NodeType.INPUT:
            return "input"
        if self.ntype == NodeType.LOGITS:
            return "logits"
        if self.ntype == NodeType.HEAD:
            return f"a{self.layer}.h{self.head}"
        return f"mlp{self.layer}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name

    def order(self, n_layers: int) -> float:
        """Position in the forward pass; lower runs first."""
        if self.ntype == NodeType.INPUT:
            return -1.0
        if self.ntype == NodeType.LOGITS:
            return 2.0 * n_layers
        sub = SUBLAYER_MLP if self.ntype == NodeType.MLP else SUBLAYER_ATTN
        return self.layer * 2 + sub

    # --- hook points -------------------------------------------------------
    def output_hook(self) -> str:
        """Hook whose activation is what this node *writes* to the residual."""
        if self.ntype == NodeType.INPUT:
            return "blocks.0.hook_resid_pre"
        if self.ntype == NodeType.HEAD:
            return f"blocks.{self.layer}.attn.hook_result"
        if self.ntype == NodeType.MLP:
            return f"blocks.{self.layer}.hook_mlp_out"
        raise ValueError(f"{self.ntype} is not a source node")

    def input_hooks(self) -> list[str]:
        """Hook(s) whose gradient is the metric's sensitivity to what this node *reads*."""
        if self.ntype == NodeType.HEAD:
            return [
                f"blocks.{self.layer}.hook_q_input",
                f"blocks.{self.layer}.hook_k_input",
                f"blocks.{self.layer}.hook_v_input",
            ]
        if self.ntype == NodeType.MLP:
            return [f"blocks.{self.layer}.hook_mlp_in"]
        if self.ntype == NodeType.LOGITS:
            # gradient w.r.t. the final residual stream
            return ["RESID_FINAL"]  # resolved by the model wrapper
        raise ValueError(f"{self.ntype} is not a destination node")


# Edge sub-types distinguish the q/k/v read paths into a head.
QKV = ("q", "k", "v")


@dataclass(frozen=True)
class Edge:
    src: Node
    dst: Node
    qkv: str = ""   # "q"/"k"/"v" when dst is a head, else ""

    @property
    def name(self) -> str:
        if self.qkv:
            return f"{self.src.name}->{self.dst.name}<{self.qkv}>"
        return f"{self.src.name}->{self.dst.name}"

    def __str__(self) -> str:  # pragma: no cover
        return self.name


class ComputationalGraph:
    """All nodes and valid residual-stream edges for an ``n_layers x n_heads`` model."""

    def __init__(self, n_layers: int, n_heads: int):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.sources: list[Node] = self._build_sources()
        self.dests: list[Node] = self._build_dests()
        self.edges: list[Edge] = self._build_edges()

    def _build_sources(self) -> list[Node]:
        nodes = [Node(NodeType.INPUT)]
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                nodes.append(Node(NodeType.HEAD, layer, head))
            nodes.append(Node(NodeType.MLP, layer))
        return nodes

    def _build_dests(self) -> list[Node]:
        nodes: list[Node] = []
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                nodes.append(Node(NodeType.HEAD, layer, head))
            nodes.append(Node(NodeType.MLP, layer))
        nodes.append(Node(NodeType.LOGITS))
        return nodes

    def _build_edges(self) -> list[Edge]:
        edges: list[Edge] = []
        for src in self.sources:
            so = src.order(self.n_layers)
            for dst in self.dests:
                if dst.order(self.n_layers) <= so:
                    continue
                if dst.ntype == NodeType.HEAD:
                    for q in QKV:
                        edges.append(Edge(src, dst, q))
                else:
                    edges.append(Edge(src, dst))
        return edges

    @property
    def all_nodes(self) -> list[Node]:
        out = list(self.sources)
        out.append(Node(NodeType.LOGITS))
        return out

    def __iter__(self) -> Iterator[Edge]:
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)

    def summary(self) -> str:
        return (
            f"ComputationalGraph(n_layers={self.n_layers}, n_heads={self.n_heads}): "
            f"{len(self.sources)} sources, {len(self.dests)} destinations, "
            f"{len(self.edges)} edges"
        )
