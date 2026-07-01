"""Ground-truth benchmark: does automated discovery find the *known* circuits?

Faithfulness says a discovered circuit reproduces the behavior; it does not say
the circuit matches what careful manual analysis found. This module scores
circuitscope's output against the published circuits:

* **IOI** (Wang et al., 2022): the 26-head circuit for GPT-2 small, by class
  (name movers, S-inhibition, induction, duplicate-token, previous-token,
  negative/backup name movers).
* **greater-than** (Hanna et al., 2023): attention heads a5.h1/a5.h5/a6.h1/
  a6.h9/a7.h10/a8.h11/a9.h1 plus late MLPs 8-11 (consensus reading of the
  paper; treated as approximate).

For each behavior we rank nodes two independent ways -- (a) exact node-level
activation patching, (b) EAP incident-edge mass -- take the top-k (k = size of
the ground-truth set), and report precision / recall / F1 plus exactly which
known components were found and missed. This turns "the circuit looks right"
into a number.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from circuitscope.behaviors import get_behavior
from circuitscope.eap import compute_eap_scores
from circuitscope.model import PatchableModel
from circuitscope.patching import baseline_metrics, patch_nodes

# --------------------------------------------------------------------------
# Published ground truth (GPT-2 small)
# --------------------------------------------------------------------------

IOI_GROUND_TRUTH: dict[str, list[str]] = {
    # Wang et al. (2022), Table 2 head classes
    "name_mover": ["a9.h9", "a10.h0", "a9.h6"],
    "negative_name_mover": ["a10.h7", "a11.h10"],
    "s_inhibition": ["a7.h3", "a7.h9", "a8.h6", "a8.h10"],
    "induction": ["a5.h5", "a5.h8", "a5.h9", "a6.h9"],
    "duplicate_token": ["a0.h1", "a0.h10", "a3.h0"],
    "previous_token": ["a2.h2", "a4.h11"],
    "backup_name_mover": ["a9.h0", "a9.h7", "a10.h1", "a10.h2",
                          "a10.h6", "a10.h10", "a11.h2", "a11.h9"],
}

GREATER_THAN_GROUND_TRUTH: dict[str, list[str]] = {
    # Hanna et al. (2023); MLP set treated as approximate consensus
    "attention": ["a5.h1", "a5.h5", "a6.h1", "a6.h9", "a7.h10", "a8.h11", "a9.h1"],
    "mlp": ["mlp8", "mlp9", "mlp10", "mlp11"],
}

GROUND_TRUTH: dict[str, dict[str, list[str]]] = {
    "ioi": IOI_GROUND_TRUTH,
    "greater_than": GREATER_THAN_GROUND_TRUTH,
}


@dataclass
class BenchmarkRow:
    behavior: str
    method: str                    # "node_patching" | "eap_edge_mass"
    k: int
    precision: float
    recall: float
    f1: float
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    recall_by_class: dict[str, float] = field(default_factory=dict)


def _flatten(gt: dict[str, list[str]]) -> set[str]:
    return {n for nodes in gt.values() for n in nodes}


def _score(ranked: list[str], gt: dict[str, list[str]], behavior: str,
           method: str, k: int | None = None) -> BenchmarkRow:
    gt_all = _flatten(gt)
    k = k or len(gt_all)
    top = set(ranked[:k])
    found = sorted(top & gt_all)
    missed = sorted(gt_all - top)
    precision = len(found) / max(1, len(top))
    recall = len(found) / max(1, len(gt_all))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    by_class = {
        cls: round(len(set(nodes) & top) / len(nodes), 3)
        for cls, nodes in gt.items()
    }
    return BenchmarkRow(behavior, method, k, round(precision, 3), round(recall, 3),
                        round(f1, 3), found, missed, by_class)


def _eap_node_mass(eap, restrict_prefix: tuple[str, ...]) -> list[str]:
    """Rank nodes by total |EAP score| over incident edges."""
    mass: dict[str, float] = {}
    for e in eap.edges:
        s = abs(eap.edge_scores[e.name])
        for name in (e.src.name, e.dst.name):
            if name.startswith(restrict_prefix):
                mass[name] = mass.get(name, 0.0) + s
    return [n for n, _ in sorted(mass.items(), key=lambda kv: -kv[1])]


def run_benchmark(
    model_name: str = "gpt2",
    behaviors: list[str] | None = None,
    n_examples: int = 8,
    device: str | None = None,
    out_dir: str | Path = "outputs",
    log=print,
) -> list[BenchmarkRow]:
    behaviors = behaviors or list(GROUND_TRUTH)
    model = PatchableModel(model_name, device=device)
    rows: list[BenchmarkRow] = []
    t0 = time.time()

    for bname in behaviors:
        gt = GROUND_TRUTH[bname]
        gt_all = _flatten(gt)
        # heads-only ground truth -> rank heads only; mixed -> rank heads+mlps
        prefixes = ("a",) if all(n.startswith("a") for n in gt_all) else ("a", "mlp")

        log(f"[benchmark] {bname}: {len(gt_all)} ground-truth components "
            f"({', '.join(gt)})")
        behavior = get_behavior(bname, n=n_examples).tokenize(model).to(model.device)
        base = baseline_metrics(model, behavior)
        log(f"[benchmark]   clean={base['clean']:.2f} corrupt={base['corrupt']:.2f}")

        # method (a): exact node activation patching
        ni = patch_nodes(model, behavior)
        ranked_np = [n for n, _ in sorted(ni.items(), key=lambda kv: -abs(kv[1]))
                     if n.startswith(prefixes)]
        rows.append(_score(ranked_np, gt, bname, "node_patching"))

        # method (b): EAP incident-edge mass
        eap = compute_eap_scores(model, behavior)
        ranked_eap = _eap_node_mass(eap, prefixes)
        rows.append(_score(ranked_eap, gt, bname, "eap_edge_mass"))

        for r in rows[-2:]:
            log(f"[benchmark]   {r.method:14s} P@{r.k}={r.precision:.2f} "
                f"R@{r.k}={r.recall:.2f} F1={r.f1:.2f}  missed={r.missed}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"benchmark_{model_name.replace('/', '_')}.json"
    path.write_text(json.dumps([vars(r) for r in rows], indent=2))
    log(f"[benchmark] wrote {path}  ({time.time() - t0:.0f}s)")
    return rows


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="circuitscope.benchmark",
        description="Score discovered circuits against published ground truth.",
    )
    p.add_argument("--model", default="gpt2")
    p.add_argument("--behaviors", default=None,
                   help=f"comma-separated subset of {list(GROUND_TRUTH)}")
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="outputs")
    args = p.parse_args(argv)
    run_benchmark(
        model_name=args.model,
        behaviors=args.behaviors.split(",") if args.behaviors else None,
        n_examples=args.examples,
        device=args.device,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
