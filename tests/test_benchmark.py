"""Fast unit tests for benchmark scoring (no model needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitscope.benchmark import GROUND_TRUTH, _flatten, _score


def test_ground_truth_shapes():
    ioi = _flatten(GROUND_TRUTH["ioi"])
    assert len(ioi) == 26                      # Wang et al. 26-head circuit
    assert "a9.h9" in ioi and "a10.h7" in ioi
    gt = _flatten(GROUND_TRUTH["greater_than"])
    assert "a9.h1" in gt and "mlp10" in gt


def test_score_perfect_and_partial():
    gt = {"x": ["a1.h1", "a2.h2"], "y": ["mlp3"]}
    # perfect: top-3 is exactly the ground truth
    r = _score(["a1.h1", "mlp3", "a2.h2", "a9.h9"], gt, "toy", "m")
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0
    assert r.missed == []
    # partial: one of three found
    r2 = _score(["a1.h1", "a5.h5", "a6.h6"], gt, "toy", "m")
    assert r2.recall == round(1 / 3, 3)
    assert set(r2.missed) == {"a2.h2", "mlp3"}
    assert r2.recall_by_class == {"x": 0.5, "y": 0.0}


def test_score_empty_ranking():
    gt = {"x": ["a1.h1"]}
    r = _score([], gt, "toy", "m")
    assert r.recall == 0.0 and r.f1 == 0.0
