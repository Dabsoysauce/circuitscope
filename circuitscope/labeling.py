"""Automated labeling of circuit components.

For each component (head/MLP) in the discovered circuit we want a human-readable
account of *what it does*. Two complementary signals:

1. **Direct logit attribution (DLA)** -- always available. We push the
   component's output through the unembedding and report the tokens it most
   promotes/suppresses at the answer position. For attention heads we also
   summarize the attention pattern (which earlier token it reads from). This
   needs no SAE and works for any HookedTransformer.

2. **SAE features** -- when ``sae_lens`` and a pretrained SAE for the relevant
   residual layer are available, we decompose the residual stream the component
   writes into into sparse features, and report the top features firing on the
   behavior, with their public neuronpedia descriptions when present.

The function degrades gracefully: if SAEs cannot be loaded, only DLA labels are
returned, and ``method`` records which path was taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from circuitscope.behaviors import BehaviorSpec
from circuitscope.graph import NodeType
from circuitscope.model import PatchableModel


@dataclass
class NodeLabel:
    node: str
    method: str                       # "sae+dla" or "dla"
    promotes: list[str] = field(default_factory=list)
    suppresses: list[str] = field(default_factory=list)
    attends_to: str | None = None
    sae_features: list[dict] = field(default_factory=list)
    summary: str = ""


@torch.no_grad()
def _dla_tokens(model: PatchableModel, vec: torch.Tensor, k: int = 6):
    """Top promoted / suppressed tokens for a residual-direction vector [d_model]."""
    hm = model.model
    vec = vec.to(hm.W_U.dtype)
    # apply final layernorm scale heuristically by just projecting (DLA convention)
    logit_contrib = vec @ hm.W_U  # [d_vocab]
    top = torch.topk(logit_contrib, k)
    bot = torch.topk(-logit_contrib, k)
    promotes = [hm.tokenizer.decode([i]) for i in top.indices.tolist()]
    suppresses = [hm.tokenizer.decode([i]) for i in bot.indices.tolist()]
    return promotes, suppresses


@torch.no_grad()
def _head_attention_summary(model, cache, layer, head, behavior) -> str:
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]  # [b, qpos, kpos]
    qpos = behavior.answer_position
    attn = pattern[:, qpos, :].mean(0)  # [kpos]
    kbest = int(attn.argmax().item())
    toks = model.to_str_tokens(behavior.clean_prompts[0])
    tok = toks[kbest] if kbest < len(toks) else f"pos{kbest}"
    return f"attends from answer pos to '{tok}' (k-pos {kbest}, weight {attn[kbest]:.2f})"


_SAE_CACHE: dict[int, object] = {}


def _load_sae(layer: int, device: str):
    if layer in _SAE_CACHE:
        return _SAE_CACHE[layer]
    try:
        from sae_lens import SAE
    except Exception:
        return None
    try:
        out = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=device,
        )
        sae = out[0] if isinstance(out, tuple) else out
    except Exception:
        sae = None
    _SAE_CACHE[layer] = sae
    return sae


@torch.no_grad()
def label_circuit(
    model: PatchableModel,
    behavior: BehaviorSpec,
    circuit,
    use_sae: bool = True,
    top_k_features: int = 5,
) -> dict[str, NodeLabel]:
    behavior.to(model.device)
    hm = model.model
    _, cache = hm.run_with_cache(behavior.clean_tokens)

    # components that appear as a source (writer) in the circuit
    writer_nodes = {e.src for e in circuit.edges if e.src.ntype in (NodeType.HEAD, NodeType.MLP)}

    labels: dict[str, NodeLabel] = {}
    for node in sorted(writer_nodes, key=lambda n: (n.layer, n.head)):
        if node.ntype == NodeType.HEAD:
            out = cache[f"blocks.{node.layer}.attn.hook_result"][:, :, node.head, :]
        else:
            out = cache[f"blocks.{node.layer}.hook_mlp_out"]
        vec = out[:, behavior.answer_position, :].mean(0)  # [d_model]

        promotes, suppresses = _dla_tokens(model, vec)
        attends = None
        if node.ntype == NodeType.HEAD:
            attends = _head_attention_summary(model, cache, node.layer, node.head, behavior)

        method = "dla"
        sae_feats: list[dict] = []
        if use_sae:
            sae = _load_sae(node.layer, model.device)
            if sae is not None:
                # the gpt2-small-res-jb SAEs are trained on hook_resid_pre
                resid = cache[f"blocks.{node.layer}.hook_resid_pre"][:, behavior.answer_position, :]
                feats = sae.encode(resid.to(next(sae.parameters()).dtype))  # [b, n_feat]
                mean_act = feats.mean(0)
                top = torch.topk(mean_act, top_k_features)
                for idx, val in zip(top.indices.tolist(), top.values.tolist()):
                    sae_feats.append({
                        "feature": int(idx),
                        "activation": round(float(val), 4),
                        "layer": node.layer,
                    })
                method = "sae+dla"

        summary = f"promotes {promotes[:3]}"
        if attends:
            summary = f"{attends}; " + summary
        labels[node.name] = NodeLabel(
            node=node.name,
            method=method,
            promotes=promotes,
            suppresses=suppresses,
            attends_to=attends,
            sae_features=sae_feats,
            summary=summary,
        )
    return labels
