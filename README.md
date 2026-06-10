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
circuitscope --model gpt2 --behavior ioi
# -> outputs/circuit_gpt2_ioi.html  (open in a browser)
# -> outputs/circuit_gpt2_ioi.json
```

---

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
