"""A small, auditable slow-flow state model for the FDG role core.

The module in this file is deliberately independent of the route controller,
the predictor, and the training loop.  It advances a right-gauge role-space
position ``M`` and, optionally, a velocity ``V`` from one- and three-month
blocks of already-mature evidence.  Only two scalar quantities are learned:
the evidence amplitude and the velocity half-life.

Evidence confidence is intentionally *not* normalised away.  A set of
time-block gradients is reduced to a unit consensus direction, a direction
coherence, and

    reliability = norm / (norm + kappa) * coherence.

The three-month reliability is the primary trust term.  Reliable one-month
evidence can veto it when the two directions disagree, but cannot amplify it:

    q = r3 * ((1 - r1) + r1 * relu(cos(D1, D3))).

The resulting innovation is signed gradient descent (minus the long-block
direction), and is bounded at the acceleration, velocity, and position
levels relative to the frozen reference core.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def _inverse_sigmoid(value: float) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError("inverse-sigmoid input must lie strictly inside (0, 1)")
    return float(math.log(float(value) / (1.0 - float(value))))


def right_gauge_projection(value: torch.Tensor) -> torch.Tensor:
    """Return ``value @ H`` without constructing the dense centring matrix.

    The final axis is the sender-role axis.  Subtracting its mean removes the
    softmax-invisible row-wise offset while leaving all other axes untouched.
    The operation works for a single matrix and for an arbitrary block of
    matrices.
    """

    if value.ndim < 2:
        raise ValueError("a role-core tensor must have at least two dimensions")
    if value.shape[-1] <= 0:
        raise ValueError("the sender-role axis must be non-empty")
    return value - value.mean(dim=-1, keepdim=True)


def safe_frobenius_cap(
    value: torch.Tensor,
    maximum: float | torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Radially cap a tensor's Frobenius norm, including the zero-bound case.

    Values already inside the ball are returned exactly (up to multiplication
    by one).  ``maximum`` may be a differentiable scalar tensor, although the
    dynamics below derives it from the reference core rather than learning it.
    """

    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if isinstance(maximum, (float, int)) and (
        not math.isfinite(float(maximum)) or float(maximum) < 0.0
    ):
        raise ValueError("maximum must be finite and non-negative")
    bound = torch.as_tensor(maximum, dtype=value.dtype, device=value.device)
    if bound.numel() != 1:
        raise ValueError("maximum must be scalar")
    bound = bound.reshape(()).clamp_min(0.0)
    norm = torch.linalg.vector_norm(value)
    scale = torch.where(
        norm > bound,
        bound / norm.clamp_min(float(eps)),
        torch.ones_like(norm),
    )
    return value * scale


@dataclass(frozen=True)
class BFlowEvidenceSummary:
    """Consensus direction and trust diagnostics for one evidence block."""

    direction: torch.Tensor
    norm: torch.Tensor
    coherence: torch.Tensor
    reliability: torch.Tensor
    count: torch.Tensor


@dataclass(frozen=True)
class BFlowState:
    """Slow role-core state in the identifiable right gauge."""

    position: torch.Tensor
    velocity: torch.Tensor


@dataclass(frozen=True)
class BFlowDiagnostics:
    """All scalar decisions and realised norm budgets for one transition."""

    mode: str
    short: BFlowEvidenceSummary
    long: BFlowEvidenceSummary
    reference_norm: torch.Tensor
    amplitude_ratio: torch.Tensor
    half_life: torch.Tensor
    retention: torch.Tensor
    cross_scale_cosine: torch.Tensor
    meta_gain: torch.Tensor
    acceleration_cap: torch.Tensor
    proposed_acceleration_norm: torch.Tensor
    acceleration_norm: torch.Tensor
    velocity_cap: torch.Tensor
    proposed_velocity_norm: torch.Tensor
    velocity_norm: torch.Tensor
    position_cap: torch.Tensor
    proposed_position_norm: torch.Tensor
    position_norm: torch.Tensor


@dataclass(frozen=True)
class BFlowOutput:
    """The effective core, next persistent state, and transition audit."""

    core: torch.Tensor
    state: BFlowState
    diagnostics: BFlowDiagnostics


def summarize_evidence_block(
    block: torch.Tensor | None,
    *,
    like: torch.Tensor,
    kappa: float | torch.Tensor,
    eps: float = 1e-12,
) -> BFlowEvidenceSummary:
    """Summarise a ``[..., receiver_role, sender_role]`` evidence block.

    Per-block directions are first right-gauge projected.  Their raw mean is
    the overall gradient ``Gamma`` and supplies both the returned direction
    and ``norm=||Gamma||``.  Coherence is the mean cosine from each non-zero
    block gradient to that overall direction, clipped to ``[0, 1]``.  Thus
    aligned blocks score one and cancelling or internally inconsistent blocks
    approach zero.  This separation makes reliability auditable and prevents
    a later unit normalisation from erasing confidence.

    ``None`` and a leading-dimension-empty tensor are valid no-evidence inputs.
    """

    if like.ndim != 2:
        raise ValueError("like must be a two-dimensional reference core")
    if like.shape[-2] <= 0 or like.shape[-1] <= 0:
        raise ValueError("reference role dimensions must be non-empty")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    kappa_tensor = torch.as_tensor(kappa, dtype=like.dtype, device=like.device)
    if kappa_tensor.numel() != 1:
        raise ValueError("kappa must be scalar")
    kappa_tensor = kappa_tensor.reshape(())
    if not bool(torch.isfinite(kappa_tensor).detach().cpu().item()) or bool(
        (kappa_tensor <= 0).detach().cpu().item()
    ):
        raise ValueError("kappa must be finite and positive")

    if block is None:
        observations = like.new_empty((0, *like.shape))
    else:
        if block.ndim < 2 or tuple(block.shape[-2:]) != tuple(like.shape):
            raise ValueError(
                "evidence must end in the reference-core shape "
                f"{tuple(like.shape)}; got {tuple(block.shape)}"
            )
        observations = block.to(dtype=like.dtype, device=like.device).reshape(
            -1, *like.shape
        )

    if observations.shape[0] == 0:
        zero = like.new_zeros(())
        return BFlowEvidenceSummary(
            direction=torch.zeros_like(like),
            norm=zero,
            coherence=zero,
            reliability=zero,
            count=zero,
        )

    projected = right_gauge_projection(observations)
    norms = torch.linalg.vector_norm(projected.reshape(projected.shape[0], -1), dim=-1)
    valid = norms > float(eps)
    valid_weight = valid.to(dtype=like.dtype)
    count = valid_weight.sum()
    safe_count = count.clamp_min(1.0)
    overall = (projected * valid_weight.view(-1, 1, 1)).sum(dim=0) / safe_count
    overall_norm = torch.linalg.vector_norm(overall)
    direction = overall / overall_norm.clamp_min(float(eps))
    has_direction = (count > 0) & (overall_norm > float(eps))
    direction = torch.where(has_direction, direction, torch.zeros_like(direction))
    unit = projected / norms.clamp_min(float(eps)).view(-1, 1, 1)
    block_cosines = (
        (unit * direction.unsqueeze(0)).reshape(projected.shape[0], -1).sum(-1)
    )
    coherence = (block_cosines * valid_weight).sum() / safe_count
    coherence = torch.where(
        has_direction,
        coherence.clamp(min=0.0, max=1.0),
        torch.zeros_like(coherence),
    )
    reliability = overall_norm / (overall_norm + kappa_tensor) * coherence
    return BFlowEvidenceSummary(
        direction=direction,
        norm=overall_norm,
        coherence=coherence,
        reliability=reliability,
        count=count,
    )


def _direction_cosine(
    first: BFlowEvidenceSummary,
    second: BFlowEvidenceSummary,
    *,
    eps: float,
) -> torch.Tensor:
    first_flat = first.direction.reshape(-1)
    second_flat = second.direction.reshape(-1)
    denominator = (
        torch.linalg.vector_norm(first_flat)
        * torch.linalg.vector_norm(second_flat)
    ).clamp_min(float(eps))
    cosine = torch.dot(first_flat, second_flat) / denominator
    has_both = (first.coherence > 0) & (second.coherence > 0)
    return torch.where(has_both, cosine.clamp(min=-1.0, max=1.0), cosine.new_zeros(()))


class BFlowDynamics(nn.Module):
    """Advance a bounded slow role-core position (and optional velocity).

    ``position_only`` implements ``M' = M + a``.  It is the matched control
    without inertial carry.  ``position_velocity`` implements
    ``V* = rho V + a`` and ``M' = M + V*``.  If the position cap truncates
    that move, the stored velocity is the *realised* displacement, keeping the
    state kinematically consistent at the boundary.

    The only learnable parameters are ``amplitude_raw`` and
    ``half_life_raw``.  Their physical values are constrained respectively to
    ``(0, acceleration_ratio)`` and ``[6, 36]`` episodes.  The half-life is
    used only by the position-velocity arm; it is retained but inert in the
    matched position-only control.
    """

    _MODES = {"position_only", "position_velocity"}

    def __init__(
        self,
        *,
        mode: str = "position_velocity",
        initial_amplitude_ratio: float = 5e-4,
        initial_half_life: float = 12.0,
        acceleration_ratio: float = 0.001,
        velocity_ratio: float = 0.005,
        position_ratio: float = 0.20,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {sorted(self._MODES)}")
        for name, value in (
            ("acceleration_ratio", acceleration_ratio),
            ("velocity_ratio", velocity_ratio),
            ("position_ratio", position_ratio),
            ("eps", eps),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < float(initial_amplitude_ratio) < float(acceleration_ratio):
            raise ValueError(
                "initial_amplitude_ratio must lie strictly between zero and "
                "acceleration_ratio"
            )
        if not 6.0 < float(initial_half_life) < 36.0:
            raise ValueError("initial_half_life must lie strictly inside (6, 36)")

        self.mode = str(mode)
        self.acceleration_ratio = float(acceleration_ratio)
        self.velocity_ratio = float(velocity_ratio)
        self.position_ratio = float(position_ratio)
        self.eps = float(eps)
        amplitude_fraction = float(initial_amplitude_ratio) / self.acceleration_ratio
        half_life_fraction = (float(initial_half_life) - 6.0) / 30.0
        self.amplitude_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(amplitude_fraction)))
        self.half_life_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(half_life_fraction)))

    @property
    def amplitude_ratio_value(self) -> torch.Tensor:
        """Current physical innovation amplitude as a fraction of ``||B0||``."""

        return self.acceleration_ratio * torch.sigmoid(self.amplitude_raw)

    @property
    def half_life_value(self) -> torch.Tensor:
        """Current decay half-life, constrained to 6--36 episode steps."""

        return 6.0 + 30.0 * torch.sigmoid(self.half_life_raw)

    def _validate_reference_and_state(
        self,
        reference_core: torch.Tensor,
        state: BFlowState | None,
    ) -> BFlowState:
        if reference_core.ndim != 2:
            raise ValueError("reference_core must be a two-dimensional role core")
        if reference_core.shape[0] <= 0 or reference_core.shape[1] <= 0:
            raise ValueError("reference_core role dimensions must be non-empty")
        if not reference_core.is_floating_point():
            raise ValueError("reference_core must have a floating-point dtype")
        if state is None:
            return BFlowState(
                position=torch.zeros_like(reference_core),
                velocity=torch.zeros_like(reference_core),
            )
        if tuple(state.position.shape) != tuple(reference_core.shape):
            raise ValueError("state.position must match reference_core")
        if tuple(state.velocity.shape) != tuple(reference_core.shape):
            raise ValueError("state.velocity must match reference_core")
        return BFlowState(
            position=state.position.to(
                dtype=reference_core.dtype, device=reference_core.device
            ),
            velocity=state.velocity.to(
                dtype=reference_core.dtype, device=reference_core.device
            ),
        )

    @staticmethod
    def _zero_summary(like: torch.Tensor) -> BFlowEvidenceSummary:
        zero = like.new_zeros(())
        return BFlowEvidenceSummary(
            direction=torch.zeros_like(like),
            norm=zero,
            coherence=zero,
            reliability=zero,
            count=zero,
        )

    @staticmethod
    def _prepare_summary(
        summary: BFlowEvidenceSummary | None,
        *,
        like: torch.Tensor,
        name: str,
    ) -> BFlowEvidenceSummary:
        if summary is None:
            return BFlowDynamics._zero_summary(like)
        if tuple(summary.direction.shape) != tuple(like.shape):
            raise ValueError(f"{name}.direction must match reference_core")
        scalars: dict[str, torch.Tensor] = {}
        for field in ("norm", "coherence", "reliability", "count"):
            value = getattr(summary, field)
            if value.numel() != 1:
                raise ValueError(f"{name}.{field} must be scalar")
            scalars[field] = value.to(dtype=like.dtype, device=like.device).reshape(())
        return BFlowEvidenceSummary(
            direction=summary.direction.to(dtype=like.dtype, device=like.device),
            norm=scalars["norm"],
            coherence=scalars["coherence"],
            reliability=scalars["reliability"],
            count=scalars["count"],
        )

    def forward(
        self,
        reference_core: torch.Tensor,
        state: BFlowState | None = None,
        *,
        short: BFlowEvidenceSummary | None = None,
        long: BFlowEvidenceSummary | None = None,
    ) -> BFlowOutput:
        """Advance one episode from pre-calibrated short/long summaries.

        The caller deliberately computes the two summaries separately, using
        the scale calibration appropriate to each horizon (``kappa1`` and
        ``kappa3``).  This module consumes their reliability values verbatim;
        it neither recalibrates nor renormalises them.
        """

        state = self._validate_reference_and_state(reference_core, state)
        reference_norm = torch.linalg.vector_norm(reference_core)
        acceleration_cap = self.acceleration_ratio * reference_norm
        velocity_cap = self.velocity_ratio * reference_norm
        position_cap = self.position_ratio * reference_norm

        prior_position = safe_frobenius_cap(
            right_gauge_projection(state.position), position_cap, eps=self.eps
        )
        prior_velocity = safe_frobenius_cap(
            right_gauge_projection(state.velocity), velocity_cap, eps=self.eps
        )
        short = self._prepare_summary(short, like=reference_core, name="short")
        long = self._prepare_summary(long, like=reference_core, name="long")
        cross_scale_cosine = _direction_cosine(short, long, eps=self.eps)
        # A reliable short block only confirms or vetoes the long signal.  It
        # never creates an update without long evidence and never boosts q
        # above r3.  Crucially, q is not normalised after this calculation.
        meta_gain = long.reliability * (
            (1.0 - short.reliability)
            + short.reliability * torch.relu(cross_scale_cosine)
        )

        amplitude_ratio = self.amplitude_ratio_value.to(
            dtype=reference_core.dtype, device=reference_core.device
        )
        half_life = self.half_life_value.to(
            dtype=reference_core.dtype, device=reference_core.device
        )
        retention = torch.exp(reference_core.new_tensor(math.log(0.5)) / half_life)
        proposed_acceleration = (
            -amplitude_ratio * reference_norm * meta_gain * long.direction
        )
        acceleration = safe_frobenius_cap(
            right_gauge_projection(proposed_acceleration),
            acceleration_cap,
            eps=self.eps,
        )

        if self.mode == "position_velocity":
            proposed_velocity = retention * prior_velocity + acceleration
            candidate_velocity = safe_frobenius_cap(
                right_gauge_projection(proposed_velocity), velocity_cap, eps=self.eps
            )
            proposed_position = prior_position + candidate_velocity
            position = safe_frobenius_cap(
                right_gauge_projection(proposed_position), position_cap, eps=self.eps
            )
            # Projection onto a convex norm ball cannot increase displacement
            # from an already feasible prior.  Store that realised displacement
            # rather than momentum that the position cap prevented.
            velocity = right_gauge_projection(position - prior_position)
            velocity = safe_frobenius_cap(velocity, velocity_cap, eps=self.eps)
        else:
            proposed_velocity = torch.zeros_like(reference_core)
            # The position-only arm receives exactly the same acceleration as
            # the inertial arm.  It differs only by omitting velocity carry;
            # half-life therefore has no effect in this matched control.
            proposed_position = prior_position + acceleration
            position = safe_frobenius_cap(
                right_gauge_projection(proposed_position), position_cap, eps=self.eps
            )
            velocity = torch.zeros_like(reference_core)

        diagnostics = BFlowDiagnostics(
            mode=self.mode,
            short=short,
            long=long,
            reference_norm=reference_norm,
            amplitude_ratio=amplitude_ratio,
            half_life=half_life,
            retention=retention,
            cross_scale_cosine=cross_scale_cosine,
            meta_gain=meta_gain,
            acceleration_cap=acceleration_cap,
            proposed_acceleration_norm=torch.linalg.vector_norm(proposed_acceleration),
            acceleration_norm=torch.linalg.vector_norm(acceleration),
            velocity_cap=velocity_cap,
            proposed_velocity_norm=torch.linalg.vector_norm(proposed_velocity),
            velocity_norm=torch.linalg.vector_norm(velocity),
            position_cap=position_cap,
            proposed_position_norm=torch.linalg.vector_norm(proposed_position),
            position_norm=torch.linalg.vector_norm(position),
        )
        return BFlowOutput(
            core=reference_core + position,
            state=BFlowState(position=position, velocity=velocity),
            diagnostics=diagnostics,
        )
