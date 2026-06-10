"""End-to-end pipeline: model + behavior -> validated, labeled, drawn circuit."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from circuitscope.acdc import Circuit, discover_circuit
from circuitscope.behaviors import BehaviorSpec, get_behavior
from circuitscope.eap import EAPResult, compute_eap_scores
from circuitscope.labeling import label_circuit
from circuitscope.model import PatchableModel
from circuitscope.patching import patch_nodes
from circuitscope.viz import circuit_to_dict, render_html


@dataclass
class PipelineResult:
    circuit: Circuit
    eap: EAPResult
    labels: dict
    html_path: Path | None
    json_path: Path | None
    elapsed: float


def run_pipeline(
    model_name: str = "gpt2",
    behavior_name: str = "ioi",
    n_examples: int = 8,
    target_faithfulness: float = 0.7,
    max_edges: int | None = None,
    use_sae: bool = True,
    node_patching: bool = True,
    device: str | None = None,
    out_dir: str | Path = "outputs",
    behavior: BehaviorSpec | None = None,
    log=print,
) -> PipelineResult:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log(f"[1/6] loading model '{model_name}' ...")
    model = PatchableModel(model_name, device=device)
    log(f"      device={model.device}  graph={model.graph.summary()}")

    log(f"[2/6] building behavior '{behavior_name}' ...")
    behavior = behavior or get_behavior(behavior_name, n=n_examples)
    behavior.tokenize(model).to(model.device)
    log(f"      {behavior.batch_size()} prompt pairs, seq len {behavior.clean_tokens.shape[1]}")

    log("[3/6] edge attribution patching (all edges, one backward pass) ...")
    eap = compute_eap_scores(model, behavior)
    top = eap.top(8)
    log("      top edges: " + ", ".join(f"{e.name}({s:+.2f})" for e, s in top))

    log(f"[4/6] ACDC-style pruning to faithfulness >= {target_faithfulness} ...")
    circuit = discover_circuit(model, behavior, eap, target_faithfulness, max_edges)
    log(f"      circuit: {len(circuit.edges)} edges, {len(circuit.nodes)} nodes")
    log(f"      faithfulness={circuit.faithfulness:.2%}  completeness={circuit.completeness:.2%}")
    log("      faithfulness curve (edges->recovered): " +
        ", ".join(f"{n}:{f:.0%}" for n, f in circuit.faithfulness_curve))

    if node_patching:
        log("      node-level activation patching (independent causal check) ...")
        ni = patch_nodes(model, behavior)
        circuit.node_importance = ni
        top_nodes = sorted(ni.items(), key=lambda kv: -abs(kv[1]))[:8]
        log("      top causal nodes: " + ", ".join(f"{n}({v:+.2f})" for n, v in top_nodes))

    log(f"[5/6] labeling components (use_sae={use_sae}) ...")
    labels = label_circuit(model, behavior, circuit, use_sae=use_sae)
    methods = {l.method for l in labels.values()}
    log(f"      labeled {len(labels)} components via {methods or '—'}")

    log("[6/6] rendering circuit diagram ...")
    title = f"{model_name} · {behavior_name}"
    html = render_html(circuit, labels, model.n_layers, title)
    stub = f"{model_name}_{behavior_name}".replace("/", "_")
    html_path = out / f"circuit_{stub}.html"
    json_path = out / f"circuit_{stub}.json"
    html_path.write_text(html)
    payload = circuit_to_dict(circuit, labels, model.n_layers)
    payload["model"] = model_name
    payload["behavior"] = behavior_name
    payload["labels"] = {k: vars(v) for k, v in labels.items()}
    json_path.write_text(json.dumps(payload, indent=2))
    log(f"      wrote {html_path}")
    log(f"      wrote {json_path}")

    elapsed = time.time() - t0
    log(f"done in {elapsed:.1f}s")
    return PipelineResult(circuit, eap, labels, html_path, json_path, elapsed)
