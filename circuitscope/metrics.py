"""Pluggable behavior metrics: logits -> scalar, higher = behavior present.

Every stage of the pipeline (EAP attribution, ablation validation, node
patching) scores behavior through a single metric function. The right choice
depends on the behavior:

* ``logit_diff``  -- logit(correct) - logit(wrong). Linear in the logits, the
  standard faithful metric for contrastive circuit work. Default.
* ``prob_diff``   -- p(correct) - p(wrong). Bounded, useful when logit scales
  differ wildly across examples.
* ``logprob``     -- log p(correct). For behaviors with no natural wrong answer.
* ``neg_kl``      -- -KL(clean distribution || current distribution) at the
  answer position. The standard *faithfulness reporting* metric in the circuits
  literature. NOTE: its gradient vanishes exactly at the clean run (the KL
  minimum), so it is unusable for clean-run gradient attribution; when selected,
  attribution falls back to logit_diff gradients and only validation uses KL.

All functions take (behavior, logits, per_example) and return a scalar tensor
(or a per-example vector). They must stay differentiable w.r.t. logits.
"""

from __future__ import annotations

from typing import Callable

import torch


def _answer_logits(behavior, logits: torch.Tensor) -> torch.Tensor:
    """Select the [batch, vocab] logits at each example's answer position."""
    idx = torch.arange(logits.shape[0], device=logits.device)
    if behavior.answer_index is not None:
        return logits[idx, behavior.answer_index, :]
    return logits[:, behavior.answer_position, :]


def logit_diff(behavior, logits: torch.Tensor, per_example: bool = False) -> torch.Tensor:
    final = _answer_logits(behavior, logits)
    idx = torch.arange(final.shape[0], device=final.device)
    diff = final[idx, behavior.correct_ids] - final[idx, behavior.wrong_ids]
    return diff if per_example else diff.mean()


def prob_diff(behavior, logits: torch.Tensor, per_example: bool = False) -> torch.Tensor:
    final = _answer_logits(behavior, logits).softmax(dim=-1)
    idx = torch.arange(final.shape[0], device=final.device)
    diff = final[idx, behavior.correct_ids] - final[idx, behavior.wrong_ids]
    return diff if per_example else diff.mean()


def logprob(behavior, logits: torch.Tensor, per_example: bool = False) -> torch.Tensor:
    final = _answer_logits(behavior, logits).log_softmax(dim=-1)
    idx = torch.arange(final.shape[0], device=final.device)
    lp = final[idx, behavior.correct_ids]
    return lp if per_example else lp.mean()


def neg_kl(behavior, logits: torch.Tensor, per_example: bool = False) -> torch.Tensor:
    """-KL(reference || current) at the answer position; 0 iff distributions match.

    Requires ``behavior.set_reference(clean_logits)`` to have been called (the
    pipeline does this in baseline_metrics).
    """
    ref = behavior.reference_logprobs
    if ref is None:
        raise RuntimeError(
            "neg_kl metric needs a reference distribution; call "
            "behavior.set_reference(clean_logits) first."
        )
    cur = _answer_logits(behavior, logits).log_softmax(dim=-1)
    kl = (ref.exp() * (ref - cur)).sum(dim=-1)  # KL(ref || cur) per example
    out = -kl
    return out if per_example else out.mean()


METRICS: dict[str, Callable] = {
    "logit_diff": logit_diff,
    "prob_diff": prob_diff,
    "logprob": logprob,
    "neg_kl": neg_kl,
}

# metrics whose gradient is degenerate at the clean run; attribution falls back
# to logit_diff for these (validation still uses them).
GRADIENT_DEGENERATE_AT_CLEAN = {"neg_kl"}


def list_metrics() -> list[str]:
    return sorted(METRICS)
