"""Thin wrapper around HookedTransformer configured for edge-level patching.

Edge attribution patching needs the residual stream decomposed at every read
and write point. That requires three non-default flags:

* ``use_attn_result``      -> ``blocks.L.attn.hook_result`` (per-head output)
* ``use_split_qkv_input``  -> ``blocks.L.hook_{q,k,v}_input`` (per-head reads)
* ``use_hook_mlp_in``      -> ``blocks.L.hook_mlp_in`` (MLP read)

These increase memory but make the additive structure of the residual stream
explicit, which is what lets us attribute individual edges.
"""

from __future__ import annotations

import torch

from circuitscope.graph import ComputationalGraph


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PatchableModel:
    """A HookedTransformer plus its computational graph."""

    def __init__(self, model_name: str = "gpt2", device: str | None = None, dtype=torch.float32):
        from transformer_lens import HookedTransformer

        self.device = pick_device(device)
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            dtype=dtype,
        )
        self.model.set_use_attn_result(True)
        self.model.set_use_split_qkv_input(True)
        self.model.set_use_hook_mlp_in(True)
        self.model.eval()
        cfg = self.model.cfg
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.graph = ComputationalGraph(self.n_layers, self.n_heads)

    @property
    def resid_final_hook(self) -> str:
        return f"blocks.{self.n_layers - 1}.hook_resid_post"

    def to_tokens(self, *args, **kwargs):
        return self.model.to_tokens(*args, **kwargs)

    def to_str_tokens(self, *args, **kwargs):
        return self.model.to_str_tokens(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def run_with_cache(self, *args, **kwargs):
        return self.model.run_with_cache(*args, **kwargs)

    def run_with_hooks(self, *args, **kwargs):
        return self.model.run_with_hooks(*args, **kwargs)
