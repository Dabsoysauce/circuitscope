"""Edge Attribution Patching (EAP).

EAP estimates, for *every* edge in the computational graph at once, how much
that edge matters to the behavior -- using a single backward pass instead of
one forward pass per edge (which is what brute-force activation patching costs).

The idea (Nanda 2023; Syed, Rager & Conmy 2023): patching the activation that
flows along edge ``u -> d`` from its clean value to its corrupt value changes
the metric ``L`` by approximately

    score(u->d) = (a_corrupt(u) - a_clean(u)) . dL/d(input_d) | clean

i.e. the dot product (over position and model dimension) of the *change in the
source's output* with the *gradient of the metric w.r.t. the destination's
input*, both read from cached activations. A large ``|score|`` means the edge
carries behavior-relevant information. The sign tells direction: a positive
score means corrupting the edge *reduces* the metric, so the edge supports the
behavior.

This module returns a dense score for every edge, which downstream pruning
(ACDC-style) thresholds into a minimal circuit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from circuitscope.behaviors import BehaviorSpec
from circuitscope.graph import ComputationalGraph, Edge, Node, NodeType
from circuitscope.model import PatchableModel


@dataclass
class EAPResult:
    edge_scores: dict[str, float]          # edge.name -> signed score
    edges: list[Edge]                      # parallel to ordering used
    graph: ComputationalGraph

    def ranked(self) -> list[tuple[Edge, float]]:
        order = sorted(self.edges, key=lambda e: -abs(self.edge_scores[e.name]))
        return [(e, self.edge_scores[e.name]) for e in order]

    def top(self, k: int = 20) -> list[tuple[Edge, float]]:
        return self.ranked()[:k]


def _src_slice(act: torch.Tensor, node: Node) -> torch.Tensor:
    """Pick the [b, p, d_model] output that `node` writes from a hook activation."""
    if node.ntype == NodeType.HEAD:
        return act[:, :, node.head, :]
    return act  # input (resid_pre) and mlp_out are already [b, p, d_model]


def compute_eap_scores(model: PatchableModel, behavior: BehaviorSpec) -> EAPResult:
    g = model.graph
    hm = model.model
    behavior.to(model.device)

    # --- destination slots: (node, qkv) -> input hook name -----------------
    dst_slots: list[tuple[Node, str, str]] = []  # (node, qkv, hook_name)
    for node in g.dests:
        if node.ntype == NodeType.HEAD:
            for qkv, hook in zip(("q", "k", "v"), node.input_hooks()):
                dst_slots.append((node, qkv, hook))
        elif node.ntype == NodeType.MLP:
            dst_slots.append((node, "", node.input_hooks()[0]))
        elif node.ntype == NodeType.LOGITS:
            dst_slots.append((node, "", model.resid_final_hook))
    dst_hook_set = {h for _, _, h in dst_slots}

    # --- source hooks ------------------------------------------------------
    src_hook_of: dict[str, str] = {n.name: n.output_hook() for n in g.sources}
    src_hook_set = set(src_hook_of.values())

    # === Pass 1: corrupt run, cache source outputs =========================
    corrupt_src: dict[str, torch.Tensor] = {}

    def make_src_cacher(store):
        def hook(act, hook):
            store[hook.name] = act.detach()
        return hook

    fwd = [(h, make_src_cacher(corrupt_src)) for h in src_hook_set]
    with torch.no_grad():
        hm.run_with_hooks(behavior.corrupt_tokens, fwd_hooks=fwd, return_type=None)

    # === Pass 2: clean run, cache source outputs + destination gradients ===
    clean_src: dict[str, torch.Tensor] = {}
    grad_store: dict[str, torch.Tensor] = {}

    def src_cacher(act, hook):
        clean_src[hook.name] = act.detach()
        return act

    def grad_capturer(act, hook):
        # register a backward hook on this activation to grab its gradient
        def bwd(grad):
            grad_store[hook.name] = grad.detach()
        act.register_hook(bwd)
        return act

    fwd2 = [(h, src_cacher) for h in src_hook_set]
    fwd2 += [(h, grad_capturer) for h in dst_hook_set]

    hm.zero_grad(set_to_none=True)
    logits = hm.run_with_hooks(behavior.clean_tokens, fwd_hooks=fwd2, return_type="logits")
    metric = behavior.logit_diff(logits)
    metric.backward()

    batch = behavior.batch_size()

    # --- assemble dense source-delta and destination-grad tensors ----------
    # U: [n_src, b, p, d_model]   delta = corrupt - clean
    src_nodes = g.sources
    U = torch.stack([
        _src_slice(corrupt_src[src_hook_of[n.name]], n)
        - _src_slice(clean_src[src_hook_of[n.name]], n)
        for n in src_nodes
    ])  # [S, b, p, d]

    # G: [n_slot, b, p, d_model]
    def slot_grad(node: Node, hook: str) -> torch.Tensor:
        gr = grad_store[hook]
        if node.ntype == NodeType.HEAD:
            return gr[:, :, node.head, :]
        return gr

    G = torch.stack([slot_grad(node, hook) for node, _, hook in dst_slots])  # [T, b, p, d]

    # score[s, t] = sum_{b,p,d} U[s] * G[t]  / batch
    scores = torch.einsum("sbpd,tbpd->st", U, G) / batch  # [S, T]

    src_index = {n.name: i for i, n in enumerate(src_nodes)}
    slot_index = {(node.name, qkv): j for j, (node, qkv, _) in enumerate(dst_slots)}

    edge_scores: dict[str, float] = {}
    for e in g.edges:
        s = src_index[e.src.name]
        t = slot_index[(e.dst.name, e.qkv)]
        edge_scores[e.name] = float(scores[s, t].item())

    hm.zero_grad(set_to_none=True)
    return EAPResult(edge_scores=edge_scores, edges=list(g.edges), graph=g)
