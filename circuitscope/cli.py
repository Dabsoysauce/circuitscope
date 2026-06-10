"""Command-line entry point: discover a circuit for a (model, behavior) pair."""

from __future__ import annotations

import argparse

from circuitscope.behaviors import list_behaviors
from circuitscope.pipeline import run_pipeline


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="circuitscope",
        description="Automatically discover and causally validate the circuit "
                    "responsible for a behavior in a transformer LM.",
    )
    p.add_argument("--model", default="gpt2", help="HookedTransformer name (default: gpt2)")
    p.add_argument("--behavior", default="ioi",
                   help=f"behavior name; built-in: {list_behaviors()}")
    p.add_argument("--examples", type=int, default=8, help="prompt pairs to use")
    p.add_argument("--faithfulness", type=float, default=0.8,
                   help="target fraction of clean-vs-corrupt metric to recover")
    p.add_argument("--max-edges", type=int, default=None,
                   help="cap on edges considered during pruning")
    p.add_argument("--no-sae", action="store_true", help="skip SAE feature labeling")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if unset)")
    p.add_argument("--out", default="outputs", help="output directory")
    args = p.parse_args(argv)

    run_pipeline(
        model_name=args.model,
        behavior_name=args.behavior,
        n_examples=args.examples,
        target_faithfulness=args.faithfulness,
        max_edges=args.max_edges,
        use_sae=not args.no_sae,
        device=args.device,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
