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
from circuitscope.viz import (circuit_to_dict, feature_circuit_to_dict,
                              render_feature_html, render_html)


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


@dataclass
class FeaturePipelineResult:
    circuit: object               # FeatureCircuit
    html_path: Path | None
    json_path: Path | None
    elapsed: float


def run_feature_pipeline(
    model_name: str = "gpt2",
    behavior_name: str = "ioi",
    n_examples: int = 8,
    target_faithfulness: float = 0.8,
    layers: list[int] | None = None,
    include_errors: bool = True,
    max_features: int = 400,
    device: str | None = None,
    out_dir: str | Path = "outputs",
    behavior: BehaviorSpec | None = None,
    log=print,
) -> FeaturePipelineResult:
    """Discover a *sparse feature circuit*: the SAE features across layers that
    causally implement the behavior."""
    from circuitscope.feature_circuit import FeatureCircuitDiscoverer
    from circuitscope.sae_bank import SAEBank

    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log(f"[1/5] loading model '{model_name}' ...")
    model = PatchableModel(model_name, device=device)
    layers = layers if layers is not None else list(range(model.n_layers))
    log(f"      device={model.device}, decomposing layers {layers}")

    log(f"[2/5] building behavior '{behavior_name}' and loading {len(layers)} SAEs ...")
    behavior = behavior or get_behavior(behavior_name, n=n_examples)
    behavior.tokenize(model).to(model.device)
    bank = SAEBank(layers, device=model.device)
    log(f"      d_sae={bank.d_sae} per layer")

    log("[3/5] decomposing residual stream + attributing the metric to features ...")
    disc = FeatureCircuitDiscoverer(model, behavior, bank, include_errors=include_errors)
    top = [n for n in disc.attribute() if not n.is_error][:8]
    log("      top features: " + ", ".join(f"{n.name}({n.ie:+.2f})" for n in top))

    log(f"[4/5] selecting + exactly validating circuit to faithfulness >= {target_faithfulness} ...")
    fc = disc.discover(target_faithfulness=target_faithfulness, max_features=max_features)
    log(f"      {fc.n_features} features | faithfulness={fc.faithfulness:.2%} "
        f"(errors-only {fc.errors_only_baseline:.2%}), completeness={fc.completeness:.2%}")
    log("      curve (k features -> recovered): " +
        ", ".join(f"{k}:{v:.0%}" for k, v in fc.faithfulness_curve))

    log("[5/5] rendering feature-circuit diagram ...")
    title = f"{model_name} · {behavior_name} (features)"
    stub = f"{model_name}_{behavior_name}_features".replace("/", "_")
    html_path = out / f"circuit_{stub}.html"
    json_path = out / f"circuit_{stub}.json"
    html_path.write_text(render_feature_html(fc, model.n_layers, title))
    payload = feature_circuit_to_dict(fc, model.n_layers)
    payload.update({"model": model_name, "behavior": behavior_name, "layers": layers,
                    "labels": fc.labels})
    json_path.write_text(json.dumps(payload, indent=2))
    log(f"      wrote {html_path}")
    log(f"      wrote {json_path}")

    elapsed = time.time() - t0
    log(f"done in {elapsed:.1f}s")
    return FeaturePipelineResult(fc, html_path, json_path, elapsed)
