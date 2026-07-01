"""Fast unit tests for the metric registry (no model download needed)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from circuitscope.behaviors import BehaviorSpec
from circuitscope.metrics import METRICS, list_metrics


def make_behavior(logits_vocab=5) -> tuple[BehaviorSpec, torch.Tensor]:
    """Two examples, three positions, tiny vocab; answers at different positions."""
    b = BehaviorSpec(
        name="toy", description="toy", clean_prompts=["a", "b"],
        corrupt_prompts=["a", "b"], correct_answers=["x"], wrong_answers=["y"],
    )
    b.correct_ids = torch.tensor([2, 0])
    b.wrong_ids = torch.tensor([4, 1])
    b.answer_index = torch.tensor([2, 1])   # example 0 answers at pos 2, ex 1 at pos 1
    logits = torch.zeros(2, 3, logits_vocab)
    # example 0, pos 2: logit[2]=3, logit[4]=1  -> diff 2
    logits[0, 2, 2] = 3.0
    logits[0, 2, 4] = 1.0
    # example 1, pos 1: logit[0]=0.5, logit[1]=2.5 -> diff -2
    logits[1, 1, 0] = 0.5
    logits[1, 1, 1] = 2.5
    return b, logits


def test_logit_diff_uses_per_example_answer_index():
    b, logits = make_behavior()
    per = METRICS["logit_diff"](b, logits, True)
    assert per.tolist() == pytest.approx([2.0, -2.0])
    assert float(METRICS["logit_diff"](b, logits, False)) == pytest.approx(0.0)


def test_prob_diff_matches_manual_softmax():
    b, logits = make_behavior()
    per = METRICS["prob_diff"](b, logits, True)
    p0 = torch.softmax(logits[0, 2], dim=-1)
    assert float(per[0]) == pytest.approx(float(p0[2] - p0[4]))


def test_logprob():
    b, logits = make_behavior()
    per = METRICS["logprob"](b, logits, True)
    lp0 = torch.log_softmax(logits[0, 2], dim=-1)
    assert float(per[0]) == pytest.approx(float(lp0[2]))


def test_neg_kl_zero_at_reference_and_negative_elsewhere():
    b, logits = make_behavior()
    b.set_reference(logits)
    at_ref = METRICS["neg_kl"](b, logits, False)
    assert float(at_ref) == pytest.approx(0.0, abs=1e-6)
    other = logits.clone()
    other[0, 2, 2] = -3.0
    away = METRICS["neg_kl"](b, other, False)
    assert float(away) < -0.01


def test_neg_kl_requires_reference():
    b, logits = make_behavior()
    with pytest.raises(RuntimeError):
        METRICS["neg_kl"](b, logits, False)


def test_attribution_fallback_for_neg_kl():
    b, logits = make_behavior()
    b.metric_name = "neg_kl"
    # attribution_metric must not raise even without a reference: it falls back
    val = b.attribution_metric(logits)
    assert float(val) == pytest.approx(0.0)  # mean of [2, -2]


def test_registry_names():
    assert list_metrics() == ["logit_diff", "logprob", "neg_kl", "prob_diff"]
