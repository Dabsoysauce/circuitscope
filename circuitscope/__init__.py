"""circuitscope: automated circuit discovery with causal validation.

Given a HookedTransformer model and a behavior (a clean/corrupt dataset plus a
metric), circuitscope discovers the minimal computational subgraph responsible
for the behavior, validates it causally, labels its components with SAE
features, and renders an interactive circuit diagram.
"""

from circuitscope.graph import ComputationalGraph, Node, NodeType
from circuitscope.behaviors import BehaviorSpec, get_behavior, list_behaviors

__all__ = [
    "ComputationalGraph",
    "Node",
    "NodeType",
    "BehaviorSpec",
    "get_behavior",
    "list_behaviors",
]

__version__ = "0.1.0"
