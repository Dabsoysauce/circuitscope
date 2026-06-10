"""Slow smoke test for sparse feature-circuit discovery on GPT-2 IOI.

Requires torch, transformer_lens, and sae_lens (plus network access to fetch the
GPT-2 SAEs). Skipped automatically when unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
pytest.importorskip("transformer_lens")
pytest.importorskip("sae_lens")


@pytest.mark.slow
def test_feature_circuit_ioi():
    from circuitscope.model import PatchableModel
    from circuitscope.behaviors import get_behavior
    from circuitscope.sae_bank import SAEBank
    from circuitscope.feature_circuit import FeatureCircuitDiscoverer

    model = PatchableModel("gpt2", device="cpu")
    behavior = get_behavior("ioi", n=8).tokenize(model).to(model.device)
    bank = SAEBank([5, 7, 8, 9, 10, 11], device="cpu")
    disc = FeatureCircuitDiscoverer(model, behavior, bank, include_errors=True)

    nodes = disc.attribute()
    assert any(not n.is_error for n in nodes)
    # attribution should produce signed, finite indirect effects
    assert all(n.ie == n.ie for n in nodes)  # not NaN

    fc = disc.discover(target_faithfulness=0.7, max_features=200)
    # adding features must beat the errors-only baseline and approach the target
    assert fc.faithfulness > fc.errors_only_baseline
    assert fc.faithfulness >= 0.6
    assert 0.0 <= fc.completeness <= 1.0
    assert fc.n_features > 0


if __name__ == "__main__":
    test_feature_circuit_ioi()
    print("feature circuit smoke test passed")
