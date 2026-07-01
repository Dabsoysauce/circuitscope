# circuitscope

**Automated circuit discovery with causal validation for transformer language models.**

Give it a model and a *behavior* (e.g. indirect-object identification,
greater-than comparison) and it discovers the minimal computational subgraph
responsible for that behavior, validates it causally, labels the components with
SAE features, and renders an interactive circuit diagram — end to end, in one
command.

This is the kind of mechanistic-interpretability investigation that is usually
done by hand over weeks. circuitscope automates the whole pipeline. It extends
the [SAE4DLM / DLM-Scope](../SAE4DLM-CE) line of work from *labeling features* to
*discovering and validating circuits*.

```bash
pip install -e .                       # installs torch + transformer_lens
circuitscope --model gpt2 --behavior ioi                  # component circuit
circuitscope --model gpt2 --behavior ioi --mode features  # SAE-feature circuit
# -> outputs/circuit_gpt2_ioi[ _features ].html  (open in a browser)
# -> outputs/circuit_gpt2_ioi[ _features ].json
```

Two granularities of circuit:

* **`--mode components`** (default): nodes are attention heads / MLPs.
* **`--mode features`**: nodes are individual **SAE features** across all layers —
  the frontier granularity of Marks et al. (2024) / attribution graphs.

And a fully automated path — describe the behavior in English:

```bash
export ANTHROPIC_API_KEY=...       # needed at generation time only
circuitscope --describe "the model knows country -> capital facts"
```

`--describe` has Claude draft clean/corrupt prompt pairs, **causally validates
each pair against the subject model itself** (aligned token positions,
single-token answers, per-example logit-diff gap), regenerates failures, and
caches the validated dataset as JSON so it runs offline afterwards
([autospec.py](circuitscope/autospec.py)).

---

**▶ [Interactive explainer](https://dabsoysauce.github.io/circuitscope/explainer.html)** —
a six-step visual walkthrough of the whole pipeline ([docs/explainer.html](docs/explainer.html),
open locally in any browser; every number in it is a real measurement from this repo).

## What it does

```
 model + behavior
        │
        ▼
 [1] Edge Attribution Patching  ── one backward pass scores all 32k edges
        │
        ▼
 [2] ACDC-style pruning         ── minimal subgraph at a target faithfulness
        │
        ▼
 [3] Causal validation          ── faithfulness + completeness via exact ablation
        │
        ▼
 [4] Node activation patching   ── independent causal check (denoising)
        │
        ▼
 [5] Automated labeling         ── DLA tokens + attention pattern + SAE features
        │
        ▼
 [6] Interactive diagram        ── self-contained HTML + JSON
```

### 1. Edge Attribution Patching (`eap.py`)
The transformer is modeled as a graph: `input`, every attention head, every MLP,
and `logits`, connected by residual-stream edges (`graph.py`). EAP estimates how
much **every edge** matters with a *single* backward pass, instead of one forward
pass per edge:

```
score(u→d) = (a_corrupt(u) − a_clean(u)) · ∂metric/∂(input_d)
```

i.e. the change in the source's output dotted with the gradient of the metric at
the destination's input (Nanda 2023; Syed, Rager & Conmy 2023).

### 2. ACDC-style pruning (`acdc.py`)
Edges are ranked by `|score|`; a binary search finds the smallest top-`k` prefix
whose **exact** ablated forward pass recovers a target fraction of the
clean-vs-corrupt metric gap. Dangling edges (not on any `input → logits` path)
are pruned so the result is a connected mechanism. A faithfulness-vs-edges curve
is reported.

### 3 & 4. Causal validation (`patching.py`)
- **Faithfulness**: run the model with *only* the circuit's edges; ablate the
  rest to their corrupt values; measure the metric recovered. Ablations compose
  recursively (`CircuitRunner`).
- **Completeness**: ablate *only* the circuit; a complete circuit collapses the
  behavior.
- **Node activation patching**: classic denoising — splice each node's clean
  output into a corrupt run — gives an independent node-importance map.

### 5. Automated labeling (`labeling.py`)
For each component: direct logit attribution (which tokens it promotes), the
attention pattern (which earlier token a head reads), and — when `sae_lens` and a
pretrained SAE are available — the top SAE features firing on the behavior.
Degrades gracefully to DLA-only if SAEs can't be loaded.

### 6. Visualization (`viz.py`)
A dependency-free interactive HTML diagram: nodes by depth, edges weighted by
`|score|` and colored by sign, hover for the automated label. Plus a JSON export.

---

## Validation: it recovers the known IOI circuit

On GPT-2 small for indirect-object identification, both methods independently
surface the circuit from Wang et al. (2022):

| Method | Top components |
| --- | --- |
| EAP (edges → logits) | `a9.h9`, `a10.h7`, `a9.h6`, `a11.h10` (name movers / neg. name mover) |
| Node patching | `mlp0`, name movers `a9.h9`/`a10.h7`, S-inhibition `a8.h6`/`a8.h10`/`a7.h9`, induction `a5.h5` |

The automated labels confirm the mechanism: head 9.9 attends from the answer
position **to the indirect-object name** (weight ~0.86); the S-inhibition head
attends to the subject. Faithfulness ≈ 70% at the default target, completeness
≈ 100% (ablating the circuit destroys the behavior).

---

## Sparse feature circuits (`--mode features`)

Instead of "which heads/MLPs?", this asks **"which SAE features, across layers,
implement the behavior?"** ([feature_circuit.py](circuitscope/feature_circuit.py),
[sae_bank.py](circuitscope/sae_bank.py)). It:

1. decomposes `hook_resid_pre` at every layer into SAE features + an *error* term
   (using the `gpt2-small-res-jb` SAEs, 24,576 features/layer);
2. attributes the metric to each feature in one backward pass —
   `IE(feat) = (f_clean − f_corrupt) · (∂metric/∂resid · Wdec)`;
3. selects the smallest feature set and **validates it exactly**: re-runs the
   model on the corrupt input with each layer's residual set to
   `corrupt_resid + Σ_circuit (f_clean − f_corrupt)·Wdec` (+ clean error term).
   Downstream attention/MLP fully re-runs, so cross-position effects are exact —
   this is a real causal measurement, not the linear approximation;
4. estimates direct-path edges `edge(u→d) = mean(Δf_u)·(Wdec[u]·Wenc[d])`;
5. labels each feature by the tokens its decoder direction promotes.

On GPT-2 / IOI it recovers an interpretable feature circuit: **person-name
detectors** (L5/L3 → "Gerrard/Avery"), a **specific-name feature** (L11 →
"Steven"), **conjunction features** for the "X and Y" frame (L7/L8 → "and"), and
**subject-tracking** features (L7/L8 → "'s / himself"). Faithfulness reaches 80%
(over a 40% errors-only baseline), completeness 100%. Error nodes (the part the
SAE can't reconstruct) are kept as the uninterpreted remainder, per the
literature — the headline is how few *features* are needed on top of them.

## Benchmarked against published ground truth

`python -m circuitscope.benchmark` scores discovery against the *known* circuits
([benchmark.py](circuitscope/benchmark.py)) — precision/recall of the top-k ranked
nodes (k = ground-truth size) vs. Wang et al.'s 26-head IOI circuit and Hanna et
al.'s greater-than circuit. GPT-2 small, CPU, 8 prompt pairs:

| Behavior | Method | P@k | R@k | F1 |
| --- | --- | --- | --- | --- |
| ioi | EAP edge mass | 0.73 | 0.73 | 0.73 |
| ioi | node patching | 0.65 | 0.65 | 0.65 |
| greater_than | node patching | 0.64 | 0.64 | 0.64 |
| greater_than | EAP edge mass | 0.46 | 0.46 | 0.46 |

Per-class recall on IOI (EAP): **100% on every primary class** — name movers,
negative name movers, S-inhibition, induction, and duplicate-token heads. The
misses are exactly the classes direct-effect methods are known to be blind to:
previous-token heads (act via composition with induction heads, not directly on
the answer) and backup name movers (only activate when the primary heads are
ablated). Greater-than attention heads operate positionally on the year digits,
which aggregate-position patching dilutes — a known limitation, listed rather
than hidden.

## Metrics

`--metric` selects the behavior metric everywhere (attribution + validation):
`logit_diff` (default), `prob_diff`, `logprob`, and `neg_kl` — the standard
faithfulness-reporting metric. `neg_kl`'s gradient vanishes at the clean run by
definition, so when selected, attribution falls back to logit-diff gradients and
only validation uses KL ([metrics.py](circuitscope/metrics.py)).

## Built-in behaviors

| name | description |
| --- | --- |
| `ioi` | Indirect Object Identification (Wang et al., 2022) |
| `greater_than` | Year comparison (Hanna et al., 2023) |

Add your own by registering a `BehaviorSpec` (clean prompts, corrupt prompts,
correct/wrong answer tokens) via `behaviors.register_behavior`. The only
requirement is that clean and corrupt prompts align token-position-by-position.

---

## CLI

```
circuitscope --model gpt2 --behavior ioi \
    --examples 8 --faithfulness 0.7 \
    [--no-sae] [--max-edges N] [--device cpu|mps|cuda] [--out DIR]
```

Works on any `HookedTransformer` model name. Runs on CPU/MPS for GPT-2-scale
models; use `--device cuda` for larger ones.

## Python API

```python
from circuitscope.pipeline import run_pipeline
res = run_pipeline(model_name="gpt2", behavior_name="ioi", use_sae=True)
print(res.circuit.faithfulness, res.circuit.completeness)
print(res.html_path)
```

## Tests

```bash
pytest tests/test_graph.py        # fast, no model
pytest -m slow                    # downloads GPT-2, checks IOI recovery
```

## Scope & honesty

Fully automating circuit discovery for *arbitrary* behaviors in *arbitrary*
models is an open research problem. circuitscope is a working, causally-grounded
pipeline that solves it for the regime where the field has ground truth
(GPT-2-scale models, well-specified contrastive behaviors). The graph, EAP,
ablation, ACDC search, and labeling generalize to any `HookedTransformer`; the
hard, unsolved part is automatically *specifying* a clean/corrupt dataset for a
behavior described in natural language — that remains a manual (or LLM-assisted)
step via `BehaviorSpec`.

## Acknowledgements

Builds on ideas from ACDC (Conmy et al., 2023), attribution patching (Nanda
2023), EAP (Syed, Rager & Conmy 2023), the IOI circuit (Wang et al., 2022), the
greater-than circuit (Hanna et al., 2023), TransformerLens, and SAELens.
