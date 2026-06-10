"""Slow smoke test: the full pipeline recovers known IOI heads on GPT-2.

Skipped automatically if torch/transformer_lens or the model weights are
unavailable. Run with: pytest -m slow  (or just run this file directly).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("transformer_lens")


@pytest.mark.slow
def test_ioi_recovers_name_movers(tmp_path):
    from circuitscope.model import PatchableModel
    from circuitscope.behaviors import get_behavior
    from circuitscope.eap import compute_eap_scores
    from circuitscope.patching import patch_nodes, baseline_metrics

    model = PatchableModel("gpt2", device="cpu")
    behavior = get_behavior("ioi", n=8).tokenize(model).to(model.device)

    base = baseline_metrics(model, behavior)
    # the behavior must actually be present in the clean prompts
    assert base["clean"] - base["corrupt"] > 1.5

    eap = compute_eap_scores(model, behavior)
    top_edges = {e.name for e, _ in eap.top(15)}
    # head 9.9 is the flagship IOI name mover; it should write to logits
    assert "a9.h9->logits" in top_edges

    ni = patch_nodes(model, behavior)
    important = {n for n, v in sorted(ni.items(), key=lambda kv: -abs(kv[1]))[:10]}
    # name movers and an S-inhibition head should rank among the most causal nodes
    assert "a9.h9" in important or "a10.h7" in important
    assert "mlp0" in important


if __name__ == "__main__":
    test_ioi_recovers_name_movers(Path("/tmp"))
    print("smoke test passed")
