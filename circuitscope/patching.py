"""Exact causal patching: the ground truth that EAP only approximates.

Two primitives:

``CircuitRunner`` runs the model on clean tokens while *ablating* every edge that
is not in a candidate circuit -- replacing what flows along that edge with its
value from the corrupt run. Because the residual stream is additive, ablating
edge ``u -> d`` means subtracting ``(live_out_u - corrupt_out_u)`` from ``d``'s
input. Source outputs are read live during the same pass, so ablations compose
recursively (an ablated upstream node changes what downstream nodes see). This
is the exact evaluation used to measure circuit *faithfulness*.

``patch_nodes`` is classic denoising activation patching: run on corrupt tokens,
splice in a single node's *clean* output, and measure how much of the metric is
recovered. This gives a node-importance map independent of EAP, useful both as a
sanity check and for the node view of the diagram.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from circuitscope.behaviors import BehaviorSpec
from circuitscope.graph import Edge, Node, NodeType
from circuitscope.model import PatchableModel


class CircuitRunner:
    """Runs forward passes with arbitrary subsets of edges ablated."""

    def __init__(self, model: PatchableModel, behavior: BehaviorSpec):
        self.model = model
        self.hm = model.model
        self.behavior = behavior.to(model.device)
        self.g = model.graph
        self._corrupt_src = self._cache_corrupt_sources()
        # group every edge by its destination input hook + (head, qkv) target
        self._edges_by_dst_hook = self._index_edges()

    # --- setup -------------------------------------------------------------
    def _src_hook(self, node: Node) -> str:
        return node.output_hook()

    def _dst_hook(self, edge: Edge) -> str:
        if edge.dst.ntype == NodeType.HEAD:
            qkv_idx = {"q": 0, "k": 1, "v": 2}[edge.qkv]
            return edge.dst.input_hooks()[qkv_idx]
        if edge.dst.ntype == NodeType.MLP:
            return edge.dst.input_hooks()[0]
        return self.model.resid_final_hook  # logits

    def _cache_corrupt_sources(self) -> dict[str, torch.Tensor]:
        store: dict[str, torch.Tensor] = {}
        hooks = []
        seen = set()
        for n in self.g.sources:
            h = n.output_hook()
            if h in seen:
                continue
            seen.add(h)

            def cacher(act, hook):
                store[hook.name] = act.detach()
            hooks.append((h, cacher))
        with torch.no_grad():
            self.hm.run_with_hooks(self.behavior.corrupt_tokens, fwd_hooks=hooks, return_type=None)
        return store

    def _index_edges(self) -> dict[str, list[Edge]]:
        idx: dict[str, list[Edge]] = defaultdict(list)
        for e in self.g.edges:
            idx[self._dst_hook(e)].append(e)
        return idx

    def _corrupt_out(self, node: Node) -> torch.Tensor:
        act = self._corrupt_src[node.output_hook()]
        if node.ntype == NodeType.HEAD:
            return act[:, :, node.head, :]
        return act

    # --- the ablated forward pass -----------------------------------------
    @torch.no_grad()
    def run(self, kept_edges: set[str], return_logits: bool = True):
        """Run on clean tokens with every edge NOT in ``kept_edges`` ablated."""
        live_src: dict[str, torch.Tensor] = {}

        # cache live source outputs as they are produced
        src_hooks = []
        seen = set()
        for n in self.g.sources:
            h = n.output_hook()
            if h in seen:
                continue
            seen.add(h)

            def cacher(act, hook):
                live_src[hook.name] = act
            src_hooks.append((h, cacher))

        # patch destination inputs: subtract delta for each ablated incoming edge
        def make_patcher(dst_hook: str):
            edges = self._edges_by_dst_hook[dst_hook]

            def patcher(act, hook):
                for e in edges:
                    if e.name in kept_edges:
                        continue
                    live = live_src[e.src.output_hook()]
                    if e.src.ntype == NodeType.HEAD:
                        live = live[:, :, e.src.head, :]
                    delta = live - self._corrupt_out(e.src)  # [b,p,d]
                    if e.dst.ntype == NodeType.HEAD:
                        act[:, :, e.dst.head, :] = act[:, :, e.dst.head, :] - delta
                    else:
                        act = act - delta
                return act
            return patcher

        dst_hooks = [(h, make_patcher(h)) for h in self._edges_by_dst_hook]
        fwd = src_hooks + dst_hooks
        logits = self.hm.run_with_hooks(
            self.behavior.clean_tokens, fwd_hooks=fwd, return_type="logits"
        )
        if return_logits:
            return logits
        return self.behavior.logit_diff(logits)

    @torch.no_grad()
    def metric(self, kept_edges: set[str]) -> float:
        return float(self.run(kept_edges, return_logits=False).item())


@torch.no_grad()
def baseline_metrics(model: PatchableModel, behavior: BehaviorSpec) -> dict[str, float]:
    """Clean and corrupt (fully ablated) metric values -- the endpoints of recovery."""
    behavior.to(model.device)
    clean_logits = model.model(behavior.clean_tokens, return_type="logits")
    corrupt_logits = model.model(behavior.corrupt_tokens, return_type="logits")
    return {
        "clean": float(behavior.logit_diff(clean_logits).item()),
        "corrupt": float(behavior.logit_diff(corrupt_logits).item()),
    }


@torch.no_grad()
def patch_nodes(model: PatchableModel, behavior: BehaviorSpec) -> dict[str, float]:
    """Denoising activation patching: per-node fraction of metric recovered.

    Run on corrupt tokens, splice in each node's clean output one at a time, and
    measure recovery toward the clean metric. Returns node.name -> recovery in
    roughly [0, 1].
    """
    behavior.to(model.device)
    hm = model.model
    g = model.graph

    # cache clean source outputs
    clean_src: dict[str, torch.Tensor] = {}
    hooks, seen = [], set()
    for n in g.sources:
        h = n.output_hook()
        if h in seen:
            continue
        seen.add(h)

        def cacher(act, hook):
            clean_src[hook.name] = act.detach()
        hooks.append((h, cacher))
    hm.run_with_hooks(behavior.clean_tokens, fwd_hooks=hooks, return_type=None)

    base = baseline_metrics(model, behavior)
    denom = base["clean"] - base["corrupt"]
    if abs(denom) < 1e-6:
        denom = 1e-6

    results: dict[str, float] = {}
    for n in g.sources:
        if n.ntype == NodeType.INPUT:
            continue
        hook_name = n.output_hook()

        def patcher(act, hook, node=n, hn=hook_name):
            clean = clean_src[hn]
            if node.ntype == NodeType.HEAD:
                act[:, :, node.head, :] = clean[:, :, node.head, :]
            else:
                act[...] = clean
            return act

        logits = hm.run_with_hooks(
            behavior.corrupt_tokens, fwd_hooks=[(hook_name, patcher)], return_type="logits"
        )
        m = float(behavior.logit_diff(logits).item())
        results[n.name] = (m - base["corrupt"]) / denom
    return results
