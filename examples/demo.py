"""Discover circuits for both built-in behaviors and print a short report.

    python examples/demo.py            # runs ioi and greater_than on gpt2 (CPU)
    python examples/demo.py --sae      # also load SAE features (downloads SAEs)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitscope.pipeline import run_feature_pipeline, run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sae", action="store_true")
    ap.add_argument("--features", action="store_true",
                    help="also discover an SAE-feature circuit for IOI")
    args = ap.parse_args()

    if args.features:
        print("\n" + "=" * 70)
        print(f"  FEATURE CIRCUIT:  {args.model}  /  ioi")
        print("=" * 70)
        fres = run_feature_pipeline(model_name=args.model, behavior_name="ioi",
                                    device=args.device, target_faithfulness=0.8)
        fc = fres.circuit
        print(f"\n{fc.n_features} SAE features | faithfulness {fc.faithfulness:.1%} "
              f"(errors-only {fc.errors_only_baseline:.1%}) | diagram {fres.html_path}")

    for behavior in ("ioi", "greater_than"):
        print("\n" + "=" * 70)
        print(f"  DISCOVERING CIRCUIT:  {args.model}  /  {behavior}")
        print("=" * 70)
        res = run_pipeline(
            model_name=args.model,
            behavior_name=behavior,
            use_sae=args.sae,
            device=args.device,
            target_faithfulness=0.7,
        )
        c = res.circuit
        print(f"\nResult: {len(c.edges)} edges, {len(c.nodes)} nodes | "
              f"faithfulness {c.faithfulness:.1%}, completeness {c.completeness:.1%}")
        top = sorted(c.node_importance.items(), key=lambda kv: -abs(kv[1]))[:6]
        print("Most causal components:", ", ".join(f"{n} ({v:+.2f})" for n, v in top))
        print(f"Diagram: {res.html_path}")


if __name__ == "__main__":
    main()
