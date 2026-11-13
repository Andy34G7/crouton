"""Evaluation utilities."""

from typing import TYPE_CHECKING

__all__ = ["CroutonLM", "evaluate_bpb", "evaluate_lm_eval"]

if TYPE_CHECKING:
    from crouton.eval.eval_runner import CroutonLM, evaluate_bpb, evaluate_lm_eval


def __getattr__(name: str):
    if name in __all__:
        from crouton.eval import eval_runner

        return getattr(eval_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
