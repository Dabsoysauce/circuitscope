"""Natural-language behavior auto-spec.

Turns a free-text behavior description ("the model resolves pronouns by
gender", "the model knows country -> capital facts") into a validated
:class:`BehaviorSpec` -- the one step of circuit discovery that was previously
manual.

The loop:

1. **Generate**: ask Claude for N clean/corrupt prompt pairs with single-token
   answers, under explicit contrastive-dataset constraints (aligned token
   positions, minimal edit between clean and corrupt, answer must be the
   *next* token).
2. **Validate causally**: tokenize each pair and keep it only if (a) clean and
   corrupt tokenize to the same length, (b) the answer tokens are single,
   distinct tokens, and (c) the *target model itself* shows a per-example
   logit-difference gap (clean_diff - corrupt_diff) above a threshold. The LLM
   proposes; the subject model disposes.
3. **Repair**: if too few pairs survive, send the failures back to Claude and
   ask for replacements, up to ``max_rounds``.

Generated datasets are cached as JSON (keyed by description + model) so a
behavior is generated once and reused offline. Requires an Anthropic API key
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile) only on generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from circuitscope.behaviors import BehaviorSpec

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"

_SCHEMA = {
    "type": "object",
    "properties": {
        "behavior_name": {
            "type": "string",
            "description": "short_snake_case name for this behavior",
        },
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clean": {"type": "string"},
                    "corrupt": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "wrong_answer": {"type": "string"},
                },
                "required": ["clean", "corrupt", "correct_answer", "wrong_answer"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["behavior_name", "pairs"],
    "additionalProperties": False,
}

_SYSTEM = """You construct contrastive datasets for mechanistic-interpretability \
circuit discovery on small language models (like GPT-2 small). Given a behavior \
description, produce clean/corrupt prompt pairs satisfying ALL of these constraints:

1. CLEAN prompt: elicits the behavior; the very NEXT token the model should \
predict is the correct answer.
2. CORRUPT prompt: identical to the clean prompt except for the minimal edit that \
removes the behavioral signal (swap the key entity/word), so the correct answer is \
no longer implied. Clean and corrupt MUST have the same number of words in the \
same positions -- substitute words one-for-one, never insert or delete.
3. correct_answer / wrong_answer: the continuation tokens being contrasted. Each \
must start with a SINGLE common English word (or number) that the tokenizer treats \
as one token, typically prefixed by a space (e.g. " Paris"). wrong_answer is what \
the corrupt prompt pulls toward (or a plausible distractor).
4. Prompts must be simple, natural text a small model like GPT-2 can handle -- no \
rare names, no unusual formatting. Prefer very common words.
5. Vary surface details across pairs (different entities/objects) while keeping \
the SAME template structure so token positions align across the batch when possible.
6. Prompts must NOT end with trailing whitespace; answers must begin with a space \
if they continue a sentence."""


@dataclass
class AutoSpecResult:
    behavior: BehaviorSpec
    n_generated: int
    n_valid: int
    rounds: int
    per_example_gap: list[float]
    cache_path: Path | None
    from_cache: bool


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "custom_behavior"


def _cache_key(description: str, model_name: str, n: int) -> str:
    h = hashlib.sha256(f"{description}|{model_name}|{n}".encode()).hexdigest()[:16]
    return h


def _validate_pairs(model, pairs: list[dict], min_gap: float) -> tuple[list[dict], list[str], list[float]]:
    """Keep pairs the *subject model* certifies; return (valid, failure_notes, gaps)."""
    valid, failures, gaps = [], [], []
    hm = model.model
    for i, p in enumerate(pairs):
        try:
            ct = hm.to_tokens(p["clean"])
            kt = hm.to_tokens(p["corrupt"])
            if ct.shape != kt.shape:
                failures.append(
                    f"pair {i}: clean/corrupt tokenize to different lengths "
                    f"({ct.shape[1]} vs {kt.shape[1]}): {p['clean']!r} / {p['corrupt']!r}"
                )
                continue
            ca = hm.to_tokens(p["correct_answer"], prepend_bos=False)[0]
            wa = hm.to_tokens(p["wrong_answer"], prepend_bos=False)[0]
            if int(ca[0]) == int(wa[0]):
                failures.append(f"pair {i}: correct/wrong answers share first token")
                continue
            with torch.no_grad():
                lc = hm(ct, return_type="logits")[0, -1]
                lk = hm(kt, return_type="logits")[0, -1]
            clean_diff = float(lc[ca[0]] - lc[wa[0]])
            corrupt_diff = float(lk[ca[0]] - lk[wa[0]])
            gap = clean_diff - corrupt_diff
            if clean_diff <= 0:
                failures.append(
                    f"pair {i}: model does not prefer the correct answer on the clean "
                    f"prompt (logit diff {clean_diff:.2f}): {p['clean']!r}"
                )
                continue
            if gap < min_gap:
                failures.append(
                    f"pair {i}: clean-vs-corrupt gap too small ({gap:.2f} < {min_gap}): "
                    f"{p['clean']!r}"
                )
                continue
            valid.append(p)
            gaps.append(gap)
        except Exception as e:  # malformed pair should not kill the loop
            failures.append(f"pair {i}: error during validation: {e}")
    return valid, failures, gaps


def _call_claude(client, judge_model: str, description: str, n: int,
                 feedback: str | None = None) -> dict:
    user = (
        f"Behavior to isolate: {description}\n\n"
        f"Generate {n} clean/corrupt prompt pairs following every constraint."
    )
    if feedback:
        user += (
            "\n\nA previous batch was validated against the actual subject model "
            "and some pairs FAILED. Generate replacement pairs that avoid these "
            f"failure modes:\n{feedback}"
        )
    resp = client.messages.create(
        model=judge_model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def generate_behavior(
    description: str,
    model,                        # PatchableModel (already loaded)
    n_examples: int = 8,
    min_gap: float = 1.0,
    max_rounds: int = 3,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache_dir: str | Path = "behaviors_cache",
    use_cache: bool = True,
    log=print,
) -> AutoSpecResult:
    """Generate + causally validate a BehaviorSpec from a free-text description."""
    cache_dir = Path(cache_dir)
    key = _cache_key(description, model.model.cfg.model_name, n_examples)
    cache_path = cache_dir / f"behavior_{key}.json"

    if use_cache and cache_path.exists():
        data = json.loads(cache_path.read_text())
        spec = BehaviorSpec(**{k: v for k, v in data["spec"].items()})
        log(f"[autospec] loaded cached behavior '{spec.name}' from {cache_path}")
        return AutoSpecResult(
            behavior=spec, n_generated=data["n_generated"], n_valid=len(spec.clean_prompts),
            rounds=data["rounds"], per_example_gap=data["gaps"],
            cache_path=cache_path, from_cache=True,
        )

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "behavior auto-spec needs the anthropic SDK: pip install anthropic"
        ) from e
    client = anthropic.Anthropic()

    valid: list[dict] = []
    gaps: list[float] = []
    n_generated = 0
    feedback: str | None = None
    name = _slug(description)

    rounds = 0
    while len(valid) < n_examples and rounds < max_rounds:
        rounds += 1
        need = max(n_examples - len(valid) + 2, 4)  # over-ask; validation attrits
        log(f"[autospec] round {rounds}: asking {judge_model} for {need} pairs ...")
        try:
            data = _call_claude(client, judge_model, description, need, feedback)
        except anthropic.AuthenticationError as e:
            raise RuntimeError(
                "no Anthropic API credentials found; set ANTHROPIC_API_KEY or run "
                "'ant auth login' (auto-spec needs the API only at generation time; "
                "cached behaviors run offline)."
            ) from e
        pairs = data.get("pairs", [])
        name = data.get("behavior_name", name)
        n_generated += len(pairs)
        ok, failures, batch_gaps = _validate_pairs(model, pairs, min_gap)
        valid.extend(ok)
        gaps.extend(batch_gaps)
        log(f"[autospec]   {len(ok)}/{len(pairs)} pairs passed causal validation "
            f"({len(valid)}/{n_examples} total)")
        feedback = "\n".join(failures[-6:]) if failures else None

    if len(valid) < 2:
        raise RuntimeError(
            f"auto-spec failed: only {len(valid)} pairs passed validation after "
            f"{rounds} rounds. The behavior may not be present in this model, or the "
            "description may need to be more concrete."
        )
    valid = valid[:n_examples]
    gaps = gaps[:n_examples]

    spec = BehaviorSpec(
        name=name,
        description=description,
        clean_prompts=[p["clean"] for p in valid],
        corrupt_prompts=[p["corrupt"] for p in valid],
        correct_answers=[p["correct_answer"] for p in valid],
        wrong_answers=[p["wrong_answer"] for p in valid],
        answer_position=-1,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "description": description,
        "model": model.model.cfg.model_name,
        "n_generated": n_generated,
        "rounds": rounds,
        "gaps": [round(g, 3) for g in gaps],
        "spec": {
            "name": spec.name,
            "description": spec.description,
            "clean_prompts": spec.clean_prompts,
            "corrupt_prompts": spec.corrupt_prompts,
            "correct_answers": spec.correct_answers,
            "wrong_answers": spec.wrong_answers,
            "answer_position": spec.answer_position,
        },
    }, indent=2))
    log(f"[autospec] cached validated behavior '{spec.name}' -> {cache_path}")

    return AutoSpecResult(
        behavior=spec, n_generated=n_generated, n_valid=len(valid), rounds=rounds,
        per_example_gap=[round(g, 3) for g in gaps], cache_path=cache_path,
        from_cache=False,
    )
