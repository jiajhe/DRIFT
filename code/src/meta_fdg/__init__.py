"""Canonical Meta-FDG implementation for the AAAI experiment contract."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import DirectedRoleGraph, role_scores

__all__ = ["DirectedRoleGraph", "role_scores"]


def __getattr__(name: str) -> Any:
    """Keep data-only commands independent from the optional Torch runtime."""

    if name in __all__:
        from .models import DirectedRoleGraph, role_scores

        return {
            "DirectedRoleGraph": DirectedRoleGraph,
            "role_scores": role_scores,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
