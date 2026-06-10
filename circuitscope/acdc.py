"""ACDC-style pruning: from dense edge scores to a minimal faithful circuit.

The original ACDC (Conmy et al., 2023) greedily removes edges whose removal
barely changes the metric, one recursive pass over the graph. That needs one
forward pass *per edge* and is slow. We use the modern shortcut: rank all edges
once with EAP, then binary-search for the smallest top-``k`` prefix that still
recovers a target fraction of the clean-vs-corrupt metric gap. Each candidate is
checked with an *exact* ablated forward pass (:class:`CircuitRunner`), so the
final circuit's faithfulness is measured, not approximated.

Before scoring a candidate we prune dangling edges -- keeping only edges that
lie on some ``input -> ... -> logits`` path -- so the reported circuit is a
genuine connected mechanism rather than a bag of high-scoring fragments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from circuitscope.behaviors import BehaviorSpec
from circuitscope.eap import EAPResult
from circuitscope.graph import Edge
from circuitscope.model import PatchableModel
from circuitscope.patching import CircuitRunner, baseline_metrics


@dataclass
class Circuit:
    edges: list[Edge]
    edge_scores: dict[str, float]
    faithfulness: float
    completeness: float
    metric_value: float
    clean_metric: float
    corrupt_metric: float
    n_edges_considered: int
    target_faithfulness: float
    nodes: set[str] = field(default_factory=set)
    faithfulness_curve: list[tuple[int, float]] = field(default_factory=list)
    node_importance: dict[str, float] = field(default_factory=dict)

    def edge_names(self) -> set[str]:
        return {e.name for e in self.edges}


def _prune_dangling(edges: list[Edge]) -> list[Edge]:
    """Keep only edges on some input -> logits path (fixpoint)."""
    cur = list(edges)
    while True:
        fwd = {"input"}
        # forward reachability from input
        changed = True
        while changed:
            changed = False
            for e in cur:
                if e.src.name in fwd and e.dst.name not in fwd:
                    fwd.add(e.dst.name)
                    changed = True
        # backward reachability to logits
        bwd = {"logits"}
        changed = True
        while changed:
            changed = False
            for e in cur:
                if e.dst.name in bwd and e.src.name not in bwd:
                    bwd.add(e.src.name)
                    changed = True
        kept = [e for e in cur if e.src.name in fwd and e.dst.name in bwd]
        if len(kept) == len(cur):
            return kept
        cur = kept


def discover_circuit(
    model: PatchableModel,
    behavior: BehaviorSpec,
    eap: EAPResult,
    target_faithfulness: float = 0.8,
    max_edges: int | None = None,
) -> Circuit:
    runner = CircuitRunner(model, behavior)
    base = baseline_metrics(model, behavior)
    clean, corrupt = base["clean"], base["corrupt"]
    denom = clean - corrupt
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6

    ranked = [e for e, _ in eap.ranked()]
    upper = len(ranked) if max_edges is None else min(max_edges, len(ranked))

    def faithfulness_of(k: int) -> tuple[float, list[Edge]]:
        edges = _prune_dangling(ranked[:k])
        m = runner.metric({e.name for e in edges})
        return (m - corrupt) / denom, edges

    # exponential search for an upper bound that meets the target, then bisect
    k = 1
    best_edges: list[Edge] = []
    while k < upper:
        fa, edges = faithfulness_of(k)
        if fa >= target_faithfulness:
            best_edges = edges
            break
        k = min(k * 2, upper)
    else:
        fa, best_edges = faithfulness_of(upper)

    # bisect down to the smallest prefix meeting the target
    if best_edges:
        lo, hi = 1, k
        while lo < hi:
            mid = (lo + hi) // 2
            fa_mid, edges_mid = faithfulness_of(mid)
            if fa_mid >= target_faithfulness:
                hi = mid
                best_edges = edges_mid
            else:
                lo = mid + 1

    final_edges = _prune_dangling(best_edges)
    kept = {e.name for e in final_edges}
    m = runner.metric(kept)
    faithfulness = (m - corrupt) / denom

    # completeness: ablate ONLY the circuit; a complete circuit should collapse
    all_names = {e.name for e in eap.edges}
    complement_metric = runner.metric(all_names - kept)
    completeness = 1.0 - (complement_metric - corrupt) / denom

    nodes = set()
    for e in final_edges:
        nodes.add(e.src.name)
        nodes.add(e.dst.name)

    # faithfulness vs. edge-budget curve, for reporting the size/fidelity tradeoff
    curve: list[tuple[int, float]] = []
    grid = sorted({1, 5, 10, 20, 40, 80, 160, 320, 640, len(final_edges), upper})
    for kk in grid:
        if kk < 1 or kk > upper:
            continue
        ce = _prune_dangling(ranked[:kk])
        cm = runner.metric({e.name for e in ce})
        curve.append((len(ce), round((cm - corrupt) / denom, 4)))

    return Circuit(
        edges=final_edges,
        edge_scores={e.name: eap.edge_scores[e.name] for e in final_edges},
        faithfulness=faithfulness,
        completeness=completeness,
        metric_value=m,
        clean_metric=clean,
        corrupt_metric=corrupt,
        n_edges_considered=len(eap.edges),
        target_faithfulness=target_faithfulness,
        nodes=nodes,
        faithfulness_curve=curve,
    )
