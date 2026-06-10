"""Sparse feature circuit discovery.

Instead of asking "which attention heads and MLPs implement this behavior?", we
ask "which *SAE features*, across all layers, implement it?" -- the granularity
of Marks et al. (2024) and Anthropic's attribution graphs.

Pipeline (mirroring the component-level one, but over features):

1. **Decompose** the residual stream at every layer into SAE features + an
   *error* term (what the SAE cannot reconstruct), on both the clean and corrupt
   runs.
2. **Attribute** the metric to each feature with one backward pass:
   ``IE(feat) = (f_clean - f_corrupt) . dmetric/df``, where ``dmetric/df`` is the
   residual-stream gradient projected through the SAE decoder. Error terms get an
   analogous score.
3. **Select & validate**: rank features by |IE| and binary-search the smallest
   set whose *exact* reconstruction restores the behavior. Validation is not an
   approximation: we re-run the model on the corrupt input and, at each layer,
   set the residual to ``corrupt_resid + sum_circuit (f_clean - f_corrupt) W_dec``
   (plus the clean error term -- error nodes are the uninterpreted remainder and
   are kept, following the literature). All downstream attention/MLP computation
   re-runs from those residuals, so cross-position effects are captured exactly.
4. **Edges** between selected features are estimated along the direct residual
   path: ``edge(u -> d) = mean(f_clean_u - f_corrupt_u) * (W_dec[u] . W_enc[d])``,
   the degree to which u's output direction is read by d's encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from circuitscope.behaviors import BehaviorSpec
from circuitscope.model import PatchableModel
from circuitscope.patching import baseline_metrics
from circuitscope.sae_bank import SAEBank


@dataclass
class FeatureNode:
    layer: int
    feature: int          # -1 means the error node for this layer
    ie: float

    @property
    def is_error(self) -> bool:
        return self.feature < 0

    @property
    def name(self) -> str:
        return f"L{self.layer}.err" if self.is_error else f"L{self.layer}.f{self.feature}"


@dataclass
class FeatureEdge:
    src: str
    dst: str
    weight: float


@dataclass
class FeatureCircuit:
    nodes: list[FeatureNode]
    edges: list[FeatureEdge]
    faithfulness: float
    completeness: float
    errors_only_baseline: float
    clean_metric: float
    corrupt_metric: float
    metric_value: float
    n_features: int
    target_faithfulness: float
    include_errors: bool
    faithfulness_curve: list[tuple[int, float]] = field(default_factory=list)
    labels: dict[str, dict] = field(default_factory=dict)


class FeatureCircuitDiscoverer:
    def __init__(self, model: PatchableModel, behavior: BehaviorSpec, bank: SAEBank,
                 include_errors: bool = True):
        self.model = model
        self.hm = model.model
        self.behavior = behavior.to(model.device)
        self.bank = bank
        self.layers = bank.layers
        self.include_errors = include_errors
        self._decompose()

    # --- step 1: decompose clean & corrupt into features + errors ----------
    @torch.no_grad()
    def _decompose(self):
        self.clean_resid, self.corrupt_resid = {}, {}
        self.f_clean, self.f_corrupt = {}, {}
        self.delta_f, self.err_delta = {}, {}

        def cache_into(store):
            def hook(act, hook):
                store[hook.name] = act.detach()
            return hook

        clean_store, corrupt_store = {}, {}
        hooks_c = [(self.bank.hook(L), cache_into(clean_store)) for L in self.layers]
        hooks_k = [(self.bank.hook(L), cache_into(corrupt_store)) for L in self.layers]
        self.hm.run_with_hooks(self.behavior.clean_tokens, fwd_hooks=hooks_c, return_type=None)
        self.hm.run_with_hooks(self.behavior.corrupt_tokens, fwd_hooks=hooks_k, return_type=None)

        for L in self.layers:
            rc = clean_store[self.bank.hook(L)]
            rk = corrupt_store[self.bank.hook(L)]
            self.clean_resid[L] = rc
            self.corrupt_resid[L] = rk
            fc = self.bank.encode(L, rc)
            fk = self.bank.encode(L, rk)
            self.f_clean[L] = fc
            self.f_corrupt[L] = fk
            self.delta_f[L] = (fc - fk)                                   # [b,p,d_sae]
            err_c = rc - self.bank.decode(L, fc)
            err_k = rk - self.bank.decode(L, fk)
            self.err_delta[L] = (err_c - err_k)                           # [b,p,d_model]

    # --- step 2: feature attribution (one backward pass) -------------------
    def attribute(self) -> list[FeatureNode]:
        grad_store: dict[str, torch.Tensor] = {}

        def grab(act, hook):
            def bwd(grad):
                grad_store[hook.name] = grad.detach()
            act.register_hook(bwd)
            return act

        hooks = [(self.bank.hook(L), grab) for L in self.layers]
        self.hm.zero_grad(set_to_none=True)
        logits = self.hm.run_with_hooks(self.behavior.clean_tokens, fwd_hooks=hooks,
                                        return_type="logits")
        self.behavior.logit_diff(logits).backward()

        nodes: list[FeatureNode] = []
        for L in self.layers:
            g = grad_store[self.bank.hook(L)]                             # [b,p,d_model]
            # dmetric/df = g @ W_dec.T ; IE = sum_bp (f_clean - f_corrupt) * dmetric/df
            dmetric_df = g @ self.bank.W_dec(L).T                          # [b,p,d_sae]
            ie = (self.delta_f[L] * dmetric_df).sum(dim=(0, 1))            # [d_sae]
            nz = torch.nonzero(self.delta_f[L].abs().sum(dim=(0, 1)) > 0).flatten()
            for fi in nz.tolist():
                nodes.append(FeatureNode(L, fi, float(ie[fi].item())))
            if self.include_errors:
                ie_err = float((self.err_delta[L] * g).sum().item())
                nodes.append(FeatureNode(L, -1, ie_err))
        self.hm.zero_grad(set_to_none=True)
        nodes.sort(key=lambda n: -abs(n.ie))
        return nodes

    # --- step 3: exact validation via residual reconstruction --------------
    @torch.no_grad()
    def _delta_for(self, L: int, feat_idx: list[int]) -> torch.Tensor:
        if not feat_idx:
            base = torch.zeros_like(self.corrupt_resid[L])
        else:
            idx = torch.tensor(feat_idx, device=self.model.device)
            df = self.delta_f[L][:, :, idx]                               # [b,p,K]
            base = df @ self.bank.W_dec(L)[idx]                           # [b,p,d_model]
        return base

    @torch.no_grad()
    def run_circuit(self, feats_by_layer: dict[int, list[int]],
                    err_layers: set[int]) -> float:
        def make_hook(L):
            delta = self._delta_for(L, feats_by_layer.get(L, []))
            if L in err_layers:
                delta = delta + self.err_delta[L]

            def hook(act, hook):
                return self.corrupt_resid[L] + delta
            return hook

        hooks = [(self.bank.hook(L), make_hook(L)) for L in self.layers]
        logits = self.hm.run_with_hooks(self.behavior.corrupt_tokens, fwd_hooks=hooks,
                                        return_type="logits")
        return float(self.behavior.logit_diff(logits).item())

    def discover(self, target_faithfulness: float = 0.8,
                 max_features: int = 400) -> FeatureCircuit:
        base = baseline_metrics(self.model, self.behavior)
        clean, corrupt = base["clean"], base["corrupt"]
        denom = clean - corrupt
        if abs(denom) < 1e-6:
            denom = 1e-6

        ranked = self.attribute()
        feature_nodes = [n for n in ranked if not n.is_error]
        err_layers = {L for L in self.layers} if self.include_errors else set()

        def recovery(feat_nodes: list[FeatureNode]) -> float:
            by_layer: dict[int, list[int]] = {}
            for n in feat_nodes:
                by_layer.setdefault(n.layer, []).append(n.feature)
            m = self.run_circuit(by_layer, err_layers)
            return (m - corrupt) / denom

        errors_only = recovery([])

        # binary search for the smallest top-k feature prefix meeting the target
        upper = min(max_features, len(feature_nodes))
        lo, hi, chosen = 1, upper, feature_nodes[:upper]
        # ensure target is reachable
        if recovery(feature_nodes[:upper]) < target_faithfulness:
            chosen = feature_nodes[:upper]
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if recovery(feature_nodes[:mid]) >= target_faithfulness:
                    hi = mid
                    chosen = feature_nodes[:mid]
                else:
                    lo = mid + 1

        kept = list(chosen)
        if self.include_errors:
            kept += [n for n in ranked if n.is_error]

        by_layer: dict[int, list[int]] = {}
        for n in chosen:
            by_layer.setdefault(n.layer, []).append(n.feature)
        m = self.run_circuit(by_layer, err_layers)
        faithfulness = (m - corrupt) / denom

        # completeness: ablate the circuit's features (clean run, remove deltas)
        completeness = self._completeness(chosen, clean, corrupt, denom)

        # faithfulness curve
        curve = []
        for k in sorted({1, 2, 5, 10, 20, 40, 80, len(chosen), upper}):
            if 1 <= k <= upper:
                curve.append((k, round(recovery(feature_nodes[:k]), 4)))

        edges = self._edges(chosen)
        labels = self._label(chosen)

        return FeatureCircuit(
            nodes=kept,
            edges=edges,
            faithfulness=faithfulness,
            completeness=completeness,
            errors_only_baseline=errors_only,
            clean_metric=clean,
            corrupt_metric=corrupt,
            metric_value=m,
            n_features=len(chosen),
            target_faithfulness=target_faithfulness,
            include_errors=self.include_errors,
            faithfulness_curve=curve,
            labels=labels,
        )

    @torch.no_grad()
    def _completeness(self, chosen, clean, corrupt, denom) -> float:
        # run on CLEAN but remove the circuit features (set them to corrupt value)
        by_layer: dict[int, list[int]] = {}
        for n in chosen:
            by_layer.setdefault(n.layer, []).append(n.feature)

        def make_hook(L):
            idx = by_layer.get(L, [])
            if not idx:
                return None
            t = torch.tensor(idx, device=self.model.device)
            delta = self.delta_f[L][:, :, t] @ self.bank.W_dec(L)[t]      # remove clean-corrupt

            def hook(act, hook):
                return act - delta
            return hook

        hooks = [(self.bank.hook(L), make_hook(L)) for L in self.layers]
        hooks = [h for h in hooks if h[1] is not None]
        logits = self.hm.run_with_hooks(self.behavior.clean_tokens, fwd_hooks=hooks,
                                        return_type="logits")
        m = float(self.behavior.logit_diff(logits).item())
        # ablating the circuit should collapse the behavior; clamp to [0,1] since
        # overshoot past the corrupt baseline still just means "fully complete".
        return float(min(1.0, max(0.0, 1.0 - (m - corrupt) / denom)))

    # --- step 4: direct-path edges between selected features ---------------
    def _edges(self, chosen: list[FeatureNode], max_edges: int = 40) -> list[FeatureEdge]:
        by_layer: dict[int, list[FeatureNode]] = {}
        for n in chosen:
            by_layer.setdefault(n.layer, []).append(n)
        edges: list[FeatureEdge] = []
        layers_sorted = sorted(by_layer)
        for i, Lu in enumerate(layers_sorted):
            for Ld in layers_sorted[i + 1:]:
                Wd_u = self.bank.W_dec(Lu)                                # [d_sae, d_model]
                We_d = self.bank.W_enc(Ld)                                # [d_model, d_sae]
                for u in by_layer[Lu]:
                    du = Wd_u[u.feature]                                   # [d_model]
                    amp = float(self.delta_f[Lu][:, :, u.feature].mean().item())
                    read = du @ We_d                                      # [d_sae]
                    for d in by_layer[Ld]:
                        w = amp * float(read[d.feature].item())
                        if abs(w) > 1e-4:
                            edges.append(FeatureEdge(u.name, d.name, round(w, 5)))
        edges.sort(key=lambda e: -abs(e.weight))
        return edges[:max_edges]

    # --- feature labels via decoder-direction DLA --------------------------
    @torch.no_grad()
    def _label(self, chosen: list[FeatureNode]) -> dict[str, dict]:
        W_U = self.hm.W_U
        tok = self.hm.tokenizer
        out: dict[str, dict] = {}
        for n in chosen:
            ids = self.bank.feature_dla(n.layer, n.feature, W_U)
            out[n.name] = {
                "layer": n.layer,
                "feature": n.feature,
                "ie": round(n.ie, 4),
                "promotes": [tok.decode([i]) for i in ids],
            }
        return out
