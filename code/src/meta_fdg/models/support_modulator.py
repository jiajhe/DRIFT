"""Support-conditioned, bounded role-space modulation primitives.

This module is deliberately independent from the graph and meta-training
pipelines.  It turns a compact support summary into three quantities that a
caller may apply to a directed role graph:

* a low-rank correction to the role-to-role core;
* a sender-role logit bias; and
* a receiver-role logit bias.

Optionally it also emits a small role-conditioned FiLM adjustment for the
*sender value* path.  It never changes the encoder, skip branch, output head,
or an arbitrary per-stock score: any such adjustment is mixed through sender
roles and consumed only by directed graph message passing.

The correction is right-centred (over its sender-role axis), which removes a
sender-softmax gauge component.  All outputs are exactly zero at
initialisation, so attaching the module to an existing FDG checkpoint is a
function-preserving change until it learns evidence for a modulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class SupportModulation:
    """A bounded support-conditioned adjustment in role space.

    ``core_delta`` has receiver rows and sender columns.  ``sender_bias`` and
    ``receiver_bias`` are intended for the corresponding role-logit vectors.
    Sender-value tensors, when enabled, have shape ``[role, feature]``.
    """

    core_delta: torch.Tensor
    sender_bias: torch.Tensor
    receiver_bias: torch.Tensor
    sender_value_scale: torch.Tensor | None = None
    sender_value_shift: torch.Tensor | None = None


class SupportRoleModulator(nn.Module):
    """Map ``[horizon, direction, statistic, role]`` support state to FDG deltas.

    The per-role encoder is shared: a role is encoded only from its own
    ``n_horizons * 2 * 4`` support summary, then common heads form low-rank
    core factors and sender/receiver biases.  The default rank is two.  The
    left core factor and both bias heads are zero-initialised, while the right
    factor remains non-zero.  Consequently every output is exactly zero at
    initialisation, but gradients can immediately update the zero heads.

    ``core_delta`` is right-centred, i.e. its sender-axis mean is zero.  Its
    Frobenius norm is clipped to ``core_delta_ratio * reference_norm``.  Each
    role bias is centred across roles and has absolute value at most
    ``max_role_bias``.
    """

    _DIRECTIONS = 2
    _STATISTICS = 4

    def __init__(
        self,
        n_horizons: int,
        n_roles: int,
        *,
        hidden_dim: int = 32,
        rank: int = 2,
        global_context_dim: int = 0,
        value_feature_dim: int | None = None,
        core_delta_ratio: float = 0.10,
        max_role_bias: float = 0.25,
        max_sender_value_scale: float = 0.05,
        max_sender_value_shift: float = 0.02,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if n_horizons <= 0:
            raise ValueError("n_horizons must be positive")
        if n_roles <= 0:
            raise ValueError("n_roles must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if rank > n_roles:
            raise ValueError("rank cannot exceed n_roles")
        if global_context_dim < 0:
            raise ValueError("global_context_dim must be non-negative")
        if value_feature_dim is not None and value_feature_dim <= 0:
            raise ValueError("value_feature_dim must be positive when enabled")
        if not math.isfinite(core_delta_ratio) or core_delta_ratio < 0:
            raise ValueError("core_delta_ratio must be finite and non-negative")
        if not math.isfinite(max_role_bias) or max_role_bias < 0:
            raise ValueError("max_role_bias must be finite and non-negative")
        if not math.isfinite(max_sender_value_scale) or max_sender_value_scale < 0:
            raise ValueError("max_sender_value_scale must be finite and non-negative")
        if not math.isfinite(max_sender_value_shift) or max_sender_value_shift < 0:
            raise ValueError("max_sender_value_shift must be finite and non-negative")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        self.n_horizons = int(n_horizons)
        self.n_roles = int(n_roles)
        self.rank = int(rank)
        self.global_context_dim = int(global_context_dim)
        self.value_feature_dim = None if value_feature_dim is None else int(value_feature_dim)
        self.core_delta_ratio = float(core_delta_ratio)
        self.max_role_bias = float(max_role_bias)
        self.max_sender_value_scale = float(max_sender_value_scale)
        self.max_sender_value_shift = float(max_sender_value_shift)
        self.eps = float(eps)

        input_dim = self.n_horizons * self._DIRECTIONS * self._STATISTICS
        # This single encoder is applied independently to every role.  Keeping
        # it shared gives equivariance to a consistent permutation of roles.
        self.role_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        if self.global_context_dim > 0:
            # A permutation-invariant regime descriptor lets every role react
            # to cross-role support evidence without giving the controller an
            # arbitrary stock-level update.  Mean and dispersion retain both
            # market-wide shift and role disagreement across horizons.
            self.global_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(2 * input_dim, self.global_context_dim),
                nn.SiLU(),
                nn.Linear(self.global_context_dim, self.global_context_dim),
                nn.SiLU(),
            )
        else:
            self.global_encoder = None
        head_dim = hidden_dim + self.global_context_dim
        self.left_factor = nn.Linear(head_dim, self.rank, bias=False)
        self.right_factor = nn.Linear(head_dim, self.rank, bias=False)
        self.sender_bias_head = nn.Linear(head_dim, 1, bias=True)
        self.receiver_bias_head = nn.Linear(head_dim, 1, bias=True)
        if self.value_feature_dim is None:
            self.sender_value_scale_head: nn.Linear | None = None
            self.sender_value_shift_head: nn.Linear | None = None
        else:
            self.sender_value_scale_head = nn.Linear(head_dim, self.value_feature_dim, bias=True)
            self.sender_value_shift_head = nn.Linear(head_dim, self.value_feature_dim, bias=True)

        # Keep the attachment function-preserving at step zero.  Initialising
        # only the left factor to zero retains a direct gradient path into it
        # through the non-zero right factor.
        nn.init.zeros_(self.left_factor.weight)
        nn.init.zeros_(self.sender_bias_head.weight)
        nn.init.zeros_(self.sender_bias_head.bias)
        nn.init.zeros_(self.receiver_bias_head.weight)
        nn.init.zeros_(self.receiver_bias_head.bias)
        if self.sender_value_scale_head is not None:
            nn.init.zeros_(self.sender_value_scale_head.weight)
            nn.init.zeros_(self.sender_value_scale_head.bias)
        if self.sender_value_shift_head is not None:
            nn.init.zeros_(self.sender_value_shift_head.weight)
            nn.init.zeros_(self.sender_value_shift_head.bias)

    def _validate_role_state(self, role_state: torch.Tensor) -> None:
        expected = (self.n_horizons, self._DIRECTIONS, self._STATISTICS, self.n_roles)
        if role_state.ndim != 4 or tuple(role_state.shape) != expected:
            raise ValueError(
                "role_state must have shape "
                f"[n_horizons, 2, 4, n_roles]={expected}; got {tuple(role_state.shape)}"
            )

    def _reference_bound(self, reference_norm: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        reference = torch.as_tensor(reference_norm, dtype=like.dtype, device=like.device).detach()
        if reference.numel() != 1:
            raise ValueError("reference_norm must be a scalar")
        reference = reference.reshape(())
        if not bool(torch.isfinite(reference).item()) or bool((reference < 0).item()):
            raise ValueError("reference_norm must be finite and non-negative")
        return reference * self.core_delta_ratio

    def _bound_frobenius(self, matrix: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(matrix)
        scale = torch.clamp(max_norm / (norm + self.eps), max=1.0)
        return matrix * scale

    def _center_and_bound_bias(self, bias: torch.Tensor) -> torch.Tensor:
        centred = bias - bias.mean(dim=-1, keepdim=True)
        if self.max_role_bias == 0:
            return torch.zeros_like(centred)
        peak = centred.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_role_bias / (peak + self.eps), max=1.0)
        return centred * scale

    @staticmethod
    def _bound_value_adjustment(raw: torch.Tensor, maximum: float) -> torch.Tensor:
        # A per-entry tanh budget, rather than a batch/role normalization,
        # leaves confidence visible to the graph and preserves role
        # permutation equivariance.
        if maximum == 0:
            return torch.zeros_like(raw)
        return float(maximum) * torch.tanh(raw)

    def forward(
        self,
        role_state: torch.Tensor,
        *,
        reference_norm: float | torch.Tensor,
    ) -> SupportModulation:
        """Return zero-init, centred and bounded role-space adjustments.

        Args:
            role_state: Exactly ``[n_horizons, 2, 4, n_roles]``.  The two
                middle axes can for example represent sender/receiver and four
                support statistics; their semantics remain caller-defined.
            reference_norm: Frobenius norm of the unmodulated core, used only
                to set the correction budget.  It is detached from gradients.
        """

        self._validate_role_state(role_state)
        parameter_dtype = self.role_encoder[0].weight.dtype
        encoded_input = role_state.permute(3, 0, 1, 2).reshape(self.n_roles, -1)
        encoded = self.role_encoder(encoded_input.to(dtype=parameter_dtype))
        if self.global_encoder is not None:
            global_input = torch.cat(
                [
                    encoded_input.mean(dim=0),
                    encoded_input.std(dim=0, unbiased=False),
                ],
                dim=0,
            ).to(dtype=parameter_dtype)
            global_context = self.global_encoder(global_input.unsqueeze(0)).expand(
                self.n_roles, -1
            )
            encoded = torch.cat([encoded, global_context], dim=-1)

        left = self.left_factor(encoded)
        right = self.right_factor(encoded)
        raw_core_delta = left @ right.transpose(-1, -2)
        # Centre only on the sender (right) role axis: Delta B @ H.
        right_centred = raw_core_delta - raw_core_delta.mean(dim=-1, keepdim=True)
        core_delta = self._bound_frobenius(
            right_centred,
            self._reference_bound(reference_norm, right_centred),
        )

        sender_bias = self._center_and_bound_bias(self.sender_bias_head(encoded).squeeze(-1))
        receiver_bias = self._center_and_bound_bias(self.receiver_bias_head(encoded).squeeze(-1))
        sender_value_scale = None
        sender_value_shift = None
        if self.sender_value_scale_head is not None:
            sender_value_scale = self._bound_value_adjustment(
                self.sender_value_scale_head(encoded),
                self.max_sender_value_scale,
            )
        if self.sender_value_shift_head is not None:
            sender_value_shift = self._bound_value_adjustment(
                self.sender_value_shift_head(encoded),
                self.max_sender_value_shift,
            )
        return SupportModulation(
            core_delta=core_delta,
            sender_bias=sender_bias,
            receiver_bias=receiver_bias,
            sender_value_scale=sender_value_scale,
            sender_value_shift=sender_value_shift,
        )
