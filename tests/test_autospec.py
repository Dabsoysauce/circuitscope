"""Tests for natural-language behavior auto-spec.

The Claude call is mocked; the *causal validation* against the real subject
model is exercised for real (slow test). Fast tests cover the pure helpers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitscope.autospec import _cache_key, _slug


def test_slug():
    assert _slug("The model knows country -> capital facts!") == \
        "the_model_knows_country_capital_facts"
    assert _slug("!!!") == "custom_behavior"


def test_cache_key_deterministic_and_distinct():
    assert _cache_key("a", "gpt2", 8) == _cache_key("a", "gpt2", 8)
    assert _cache_key("a", "gpt2", 8) != _cache_key("b", "gpt2", 8)
    assert _cache_key("a", "gpt2", 8) != _cache_key("a", "gpt2-medium", 8)


@pytest.mark.slow
def test_generate_behavior_with_mocked_llm(tmp_path, monkeypatch):
    pytest.importorskip("transformer_lens")
    import circuitscope.autospec as autospec
    from circuitscope.model import PatchableModel

    model = PatchableModel("gpt2", device="cpu")

    good = [
        {"clean": "The capital of France is", "corrupt": "The capital of Germany is",
         "correct_answer": " Paris", "wrong_answer": " Berlin"},
        {"clean": "The capital of Italy is", "corrupt": "The capital of Spain is",
         "correct_answer": " Rome", "wrong_answer": " Madrid"},
        {"clean": "The capital of Japan is", "corrupt": "The capital of China is",
         "correct_answer": " Tokyo", "wrong_answer": " Beijing"},
        # a deliberately bad pair: length mismatch -> must be filtered out
        {"clean": "Up is", "corrupt": "The opposite of down is",
         "correct_answer": " up", "wrong_answer": " down"},
    ]

    def fake_call(client, judge_model, description, n, feedback=None):
        return {"behavior_name": "capital_facts", "pairs": good}

    monkeypatch.setattr(autospec, "_call_claude", fake_call)
    # anthropic.Anthropic() must not need creds for a mocked run
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", lambda: object())

    res = autospec.generate_behavior(
        "the model knows country capital facts", model,
        n_examples=3, min_gap=1.0, cache_dir=tmp_path,
    )
    assert res.behavior.name == "capital_facts"
    assert res.n_valid == 3
    assert all(g > 1.0 for g in res.per_example_gap)
    assert res.cache_path.exists()

    # second call must hit the cache without touching the (mocked) API
    monkeypatch.setattr(autospec, "_call_claude",
                        lambda *a, **k: pytest.fail("cache miss"))
    res2 = autospec.generate_behavior(
        "the model knows country capital facts", model,
        n_examples=3, cache_dir=tmp_path,
    )
    assert res2.from_cache
    assert res2.behavior.clean_prompts == res.behavior.clean_prompts

    # the generated spec must survive tokenize() for the pipeline
    spec = res.behavior.tokenize(model)
    assert spec.clean_tokens.shape == spec.corrupt_tokens.shape


if __name__ == "__main__":
    test_slug()
    test_cache_key_deterministic_and_distinct()
    print("fast autospec tests passed")
