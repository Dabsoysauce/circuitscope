"""Command-line entry point: discover a circuit for a (model, behavior) pair."""

from __future__ import annotations

import argparse

from circuitscope.behaviors import list_behaviors
from circuitscope.pipeline import run_feature_pipeline, run_pipeline


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="circuitscope",
        description="Automatically discover and causally validate the circuit "
                    "responsible for a behavior in a transformer LM.",
    )
    p.add_argument("--model", default="gpt2", help="HookedTransformer name (default: gpt2)")
    p.add_argument("--behavior", default="ioi",
                   help=f"behavior name; built-in: {list_behaviors()}")
    p.add_argument("--describe", default=None, metavar="TEXT",
                   help="auto-spec a behavior from a natural-language description "
                        "(uses the Claude API; overrides --behavior; components mode)")
    p.add_argument("--mode", default="components", choices=["components", "features"],
                   help="'components' = head/MLP circuit (EAP+ACDC); "
                        "'features' = sparse SAE-feature circuit")
    p.add_argument("--examples", type=int, default=8, help="prompt pairs to use")
    p.add_argument("--faithfulness", type=float, default=0.8,
                   help="target fraction of clean-vs-corrupt metric to recover")
    p.add_argument("--max-edges", type=int, default=None,
                   help="[components] cap on edges considered during pruning")
    p.add_argument("--max-features", type=int, default=400,
                   help="[features] cap on features considered")
    p.add_argument("--layers", default=None,
                   help="[features] comma-separated layers to decompose (default: all)")
    p.add_argument("--no-sae", action="store_true", help="[components] skip SAE feature labeling")
    p.add_argument("--metric", default="logit_diff",
                   choices=["logit_diff", "prob_diff", "logprob", "neg_kl"],
                   help="behavior metric (neg_kl: validation only; attribution "
                        "falls back to logit_diff)")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset)")
    p.add_argument("--out", default="outputs", help="output directory")
    args = p.parse_args(argv)

    if args.mode == "features":
        layers = ([int(x) for x in args.layers.split(",")] if args.layers else None)
        run_feature_pipeline(
            model_name=args.model,
            behavior_name=args.behavior,
            n_examples=args.examples,
            target_faithfulness=args.faithfulness,
            layers=layers,
            max_features=args.max_features,
            metric=args.metric,
            device=args.device,
            out_dir=args.out,
        )
    else:
        run_pipeline(
            model_name=args.model,
            behavior_name=args.behavior,
            n_examples=args.examples,
            target_faithfulness=args.faithfulness,
            max_edges=args.max_edges,
            use_sae=not args.no_sae,
            metric=args.metric,
            describe=args.describe,
            device=args.device,
            out_dir=args.out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
