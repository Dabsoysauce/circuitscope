"""A bank of pretrained SAEs, one per residual-stream layer.

Loads the ``gpt2-small-res-jb`` SAEs (trained on ``blocks.L.hook_resid_pre``)
and exposes their encoder/decoder weights so the residual stream can be
decomposed into sparse features at every layer. This is the substrate for
feature-level circuit discovery: instead of attention heads and MLPs, the nodes
of the circuit are individual SAE features.
"""

from __future__ import annotations

import torch


class SAEBank:
    def __init__(self, layers: list[int], device: str = "cpu",
                 release: str = "gpt2-small-res-jb", dtype=torch.float32):
        from sae_lens import SAE

        self.layers = list(layers)
        self.device = device
        self.saes: dict[int, object] = {}
        for layer in self.layers:
            out = SAE.from_pretrained(
                release=release, sae_id=f"blocks.{layer}.hook_resid_pre", device=device
            )
            sae = out[0] if isinstance(out, tuple) else out
            sae = sae.to(device=device)
            self.saes[layer] = sae
        self.d_sae = self.saes[self.layers[0]].cfg.d_sae

    def hook(self, layer: int) -> str:
        return f"blocks.{layer}.hook_resid_pre"

    def encode(self, layer: int, resid: torch.Tensor) -> torch.Tensor:
        sae = self.saes[layer]
        return sae.encode(resid.to(next(sae.parameters()).dtype))

    def decode(self, layer: int, feats: torch.Tensor) -> torch.Tensor:
        return self.saes[layer].decode(feats)

    def W_dec(self, layer: int) -> torch.Tensor:    # [d_sae, d_model]
        return self.saes[layer].W_dec

    def W_enc(self, layer: int) -> torch.Tensor:    # [d_model, d_sae]
        return self.saes[layer].W_enc

    def feature_dla(self, layer: int, feature: int, W_U: torch.Tensor, k: int = 6):
        """Top tokens promoted by a feature's decoder direction (direct logit attribution)."""
        vec = self.W_dec(layer)[feature].to(W_U.dtype)        # [d_model]
        logit = vec @ W_U                                      # [d_vocab]
        top = torch.topk(logit, k).indices.tolist()
        return top
