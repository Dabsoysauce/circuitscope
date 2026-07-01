"""Behavior specifications: what circuit are we hunting for?

A :class:`BehaviorSpec` bundles everything the pipeline needs to define a
behavior causally:

* a batch of *clean* prompts that elicit the behavior,
* a batch of *corrupt* prompts that break it while keeping format/length fixed
  (so patching is well-defined token-position by token-position),
* the *answer* tokens (correct vs. counterfactual) used by the metric,
* a :func:`metric` mapping logits -> a scalar that is high when the behavior is
  present. We use the logit difference (correct - wrong), the standard faithful
  metric for circuit work.

Two canonical behaviors are built in (IOI and greater-than). New behaviors are
added by registering another :class:`BehaviorSpec` factory in ``_REGISTRY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class BehaviorSpec:
    name: str
    description: str
    clean_prompts: list[str]
    corrupt_prompts: list[str]
    # token ids per example: shape considerations handled by metric builder
    correct_answers: list[str]
    wrong_answers: list[str]
    # filled in by .tokenize(model)
    clean_tokens: Optional[torch.Tensor] = None
    corrupt_tokens: Optional[torch.Tensor] = None
    correct_ids: Optional[torch.Tensor] = None
    wrong_ids: Optional[torch.Tensor] = None
    answer_position: int = -1  # position whose logits define the answer
    answer_index: Optional[torch.Tensor] = None  # per-example last-real-token index
    metric_name: str = "logit_diff"  # see circuitscope.metrics
    reference_logprobs: Optional[torch.Tensor] = None  # for neg_kl
    extra: dict = field(default_factory=dict)

    def tokenize(self, model) -> "BehaviorSpec":
        """Tokenize prompts/answers with the model's tokenizer.

        Requires that clean and corrupt prompts share the same token length so
        position-wise patching is well-defined.
        """
        clean = model.to_tokens(self.clean_prompts)
        corrupt = model.to_tokens(self.corrupt_prompts)
        if clean.shape != corrupt.shape:
            raise ValueError(
                f"clean/corrupt token shapes differ ({clean.shape} vs {corrupt.shape}); "
                "prompts must align position-by-position for patching."
            )
        self.clean_tokens = clean
        self.corrupt_tokens = corrupt
        # answers: take the first token id of the (space-prefixed) answer string
        self.correct_ids = _answer_ids(model, self.correct_answers)
        self.wrong_ids = _answer_ids(model, self.wrong_answers)
        # per-example index of the last real (non-pad) token, where the answer
        # is predicted; robust to right-padding of ragged batches.
        pad_id = getattr(model.model.tokenizer, "pad_token_id", None)
        seq = clean.shape[1]
        if pad_id is None:
            self.answer_index = torch.full((clean.shape[0],), seq - 1)
        else:
            # largest index that is not a pad token (ignores leading BOS, which
            # equals pad for GPT-2, and excludes trailing right-padding).
            pos = torch.arange(seq).expand_as(clean)
            masked = torch.where(clean != pad_id, pos, torch.full_like(pos, -1))
            self.answer_index = masked.max(dim=1).values.clamp(min=0)
        return self

    def to(self, device) -> "BehaviorSpec":
        for attr in ("clean_tokens", "corrupt_tokens", "correct_ids", "wrong_ids",
                     "answer_index", "reference_logprobs"):
            t = getattr(self, attr)
            if t is not None:
                setattr(self, attr, t.to(device))
        return self

    # --- metrics ------------------------------------------------------------
    def metric(self, logits: torch.Tensor, per_example: bool = False) -> torch.Tensor:
        """Score the behavior with the configured metric (see circuitscope.metrics)."""
        from circuitscope.metrics import METRICS

        assert self.correct_ids is not None and self.wrong_ids is not None
        return METRICS[self.metric_name](self, logits, per_example)

    def attribution_metric(self, logits: torch.Tensor) -> torch.Tensor:
        """Metric used for gradient attribution. Falls back to logit_diff for
        metrics whose gradient vanishes at the clean run (e.g. neg_kl)."""
        from circuitscope.metrics import GRADIENT_DEGENERATE_AT_CLEAN, METRICS

        name = self.metric_name
        if name in GRADIENT_DEGENERATE_AT_CLEAN:
            name = "logit_diff"
        return METRICS[name](self, logits, False)

    def set_reference(self, clean_logits: torch.Tensor) -> None:
        """Store the clean answer-position distribution (needed by neg_kl)."""
        from circuitscope.metrics import _answer_logits

        with torch.no_grad():
            self.reference_logprobs = _answer_logits(self, clean_logits).log_softmax(-1)

    def logit_diff(self, logits: torch.Tensor, per_example: bool = False) -> torch.Tensor:
        """Logit difference (correct - wrong) at the answer position."""
        from circuitscope.metrics import logit_diff as _ld

        assert self.correct_ids is not None and self.wrong_ids is not None
        return _ld(self, logits, per_example)

    def batch_size(self) -> int:
        return len(self.clean_prompts)


def _answer_ids(model, answers: list[str]) -> torch.Tensor:
    ids = []
    for a in answers:
        toks = model.to_tokens(a, prepend_bos=False)[0]
        ids.append(int(toks[0]))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Built-in behaviors
# ---------------------------------------------------------------------------

# Single fixed template so every prompt tokenizes to the same length (required
# for position-aligned patching). All slot fillers below are single GPT-2 tokens
# (verified), so length is constant across examples.
_IOI_TEMPLATE = "When {A} and {B} went to the {PLACE}, {C} gave a {OBJECT} to"
_IOI_NAMES = ["John", "Mary", "Tom", "Mark", "Paul", "Lisa", "Anna", "Sara",
              "David", "Susan", "Karen", "Steven"]
_IOI_PLACES = ["store", "park", "school", "office"]
_IOI_OBJECTS = ["drink", "book", "ball", "ring"]


def _build_ioi(n: int = 8, seed: int = 0) -> BehaviorSpec:
    """Indirect Object Identification (Wang et al., 2022).

    Clean: "When Mary and John went to the store, John gave a drink to" -> " Mary"
    Corrupt (ABC): the repeated subject is replaced by a fresh third name, which
    removes the indirect-object signal while preserving every token position.

    Both orderings (ABBA/BABA) are sampled. The single template plus single-token
    fillers guarantee identical token length across the batch.
    """
    import random

    rng = random.Random(seed)
    clean, corrupt, correct, wrong = [], [], [], []
    for _ in range(n):
        a, b = rng.sample(_IOI_NAMES, 2)            # two distinct subjects
        if rng.random() < 0.5:
            a, b = b, a
        place = rng.choice(_IOI_PLACES)
        obj = rng.choice(_IOI_OBJECTS)
        # clean: name A repeats as the giver -> the answer is the other name, B
        clean.append(_IOI_TEMPLATE.format(A=a, B=b, C=a, PLACE=place, OBJECT=obj))
        # corrupt: replace the repeated giver with a fresh name (ABC pattern)
        c = rng.choice([x for x in _IOI_NAMES if x not in (a, b)])
        corrupt.append(_IOI_TEMPLATE.format(A=a, B=b, C=c, PLACE=place, OBJECT=obj))
        correct.append(" " + b)   # indirect object
        wrong.append(" " + a)     # subject (the distractor)
    return BehaviorSpec(
        name="ioi",
        description="Indirect Object Identification (Wang et al., 2022)",
        clean_prompts=clean,
        corrupt_prompts=corrupt,
        correct_answers=correct,
        wrong_answers=wrong,
        answer_position=-1,
    )


_GT_NOUNS = ["war", "reign", "trip", "lecture", "project", "tour", "famine", "siege"]


def _build_greater_than(n: int = 8, seed: int = 0) -> BehaviorSpec:
    """Greater-than (Hanna et al., 2023).

    Clean: "The war lasted from the year 17{XX} to the year 17" -> a two-digit
    completion that must be > XX. Corrupt sets the start year to 01 so any
    completion satisfies ">", removing the comparison signal.

    The metric uses the difference between a valid (greater) year-digit token and
    an invalid (not greater) one at the final position.
    """
    import random

    rng = random.Random(seed)
    clean, corrupt, correct, wrong = [], [], [], []
    for _ in range(n):
        noun = rng.choice(_GT_NOUNS)
        xx = rng.randint(2, 98)          # start year tens/ones, e.g. 47
        century = rng.choice([16, 17, 18, 19])
        tmpl = "The {N} lasted from the year {C}{XX} to the year {C}"
        clean.append(tmpl.format(N=noun, C=century, XX=f"{xx:02d}"))
        corrupt.append(tmpl.format(N=noun, C=century, XX="01"))
        # a valid (greater) year and an invalid (not greater) year, two digits
        valid = min(xx + 1, 99)
        invalid = max(xx - 1, 0)
        correct.append(f"{valid:02d}")
        wrong.append(f"{invalid:02d}")
    return BehaviorSpec(
        name="greater_than",
        description="Greater-than year comparison (Hanna et al., 2023)",
        clean_prompts=clean,
        corrupt_prompts=corrupt,
        correct_answers=correct,
        wrong_answers=wrong,
        answer_position=-1,
    )


_REGISTRY: dict[str, Callable[..., BehaviorSpec]] = {
    "ioi": _build_ioi,
    "greater_than": _build_greater_than,
}


def list_behaviors() -> list[str]:
    return sorted(_REGISTRY)


def get_behavior(name: str, **kwargs) -> BehaviorSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown behavior '{name}'; available: {list_behaviors()}")
    return _REGISTRY[name](**kwargs)


def register_behavior(name: str, factory: Callable[..., BehaviorSpec]) -> None:
    _REGISTRY[name] = factory
