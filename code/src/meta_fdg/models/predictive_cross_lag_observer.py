"""Pure Phase22 predictive cross-lag observer and state algebra.

The module intentionally knows nothing about data handlers, support ridge
construction, query losses, or adjacency-KL evaluation.  A graph-side caller
computes the detached KL-feasible scale and passes it back here.  The only
trainable tensor is the full rank-eight observer matrix ``K``; the frozen
parent core and right-gauged basis are buffers, and the sole adapted object is
the one K32 core represented by position (and conditionally velocity).

Unlike Phase21, this module contains no posterior solve and no Euclidean cap.
Raw innovations and realised position/velocity are fail-closed against the
protocol safety radius; they are never rescaled to satisfy it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn

from .subspace_posterior_flow import (
    core_from_coordinates,
    operator_from_coordinates,
    rotate_basis,
    rotate_coordinates,
    validate_orthogonal_rotation,
    validate_orthonormal_basis,
)


PHASE22_ROLE_COUNT = 32
PHASE22_SUBSPACE_RANK = 8
PHASE22_BETA = 2.0 ** (-1.0 / 12.0)
PHASE22_KAPPA = 1.0 - 2.0 ** (-1.0 / 36.0)
PHASE22_SAFETY_RATIO = 0.50
PHASE22_ALGEBRA_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Phase22State:
    """Detached persistent position and realised velocity coordinates."""

    position: torch.Tensor
    velocity: torch.Tensor


@dataclass(frozen=True)
class Phase22Innovation:
    """One support-evidence innovation before graph-side KL projection."""

    evidence: torch.Tensor
    reliability: torch.Tensor
    coordinates: torch.Tensor
    operator_delta: torch.Tensor
    coordinate_norm: torch.Tensor


@dataclass(frozen=True)
class Phase22StaticOutput:
    """Reset/static observer output with its one live K32 core."""

    innovation: Phase22Innovation
    kl_scale: torch.Tensor
    realised_position: torch.Tensor
    realised_delta: torch.Tensor
    core_float64: torch.Tensor
    live_core: torch.Tensor


@dataclass(frozen=True)
class Phase22TransitionProposal:
    """Uncapped persistent proposal before adjacent-state KL projection."""

    previous_state: Phase22State
    innovation: Phase22Innovation
    beta: torch.Tensor
    kappa: torch.Tensor
    retained_velocity: torch.Tensor
    anchored_position: torch.Tensor
    proposed_position: torch.Tensor
    raw_displacement: torch.Tensor


@dataclass(frozen=True)
class Phase22TransitionDiagnostics:
    """State-side values needed to audit one realised transition."""

    beta: torch.Tensor
    kappa: torch.Tensor
    safety_bound: torch.Tensor
    previous_position_norm: torch.Tensor
    previous_velocity_norm: torch.Tensor
    innovation_norm: torch.Tensor
    retained_velocity_norm: torch.Tensor
    anchored_position_norm: torch.Tensor
    proposed_position_norm: torch.Tensor
    raw_displacement_norm: torch.Tensor
    kl_scale: torch.Tensor
    realised_position_norm: torch.Tensor
    realised_velocity_norm: torch.Tensor


@dataclass(frozen=True)
class Phase22TransitionOutput:
    """One live core plus the exact detached state stored for the next month."""

    state: Phase22State
    live_position: torch.Tensor
    live_velocity: torch.Tensor
    core_float64: torch.Tensor
    live_core: torch.Tensor
    diagnostics: Phase22TransitionDiagnostics


def _require_float64_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must use float64")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().detach().cpu().item()):
        raise FloatingPointError(f"{name} contains a non-finite value")
    return value


def _float64_scalar(
    value: float | torch.Tensor,
    *,
    like: torch.Tensor,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> torch.Tensor:
    scalar = torch.as_tensor(value, dtype=torch.float64, device=like.device)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    scalar = scalar.reshape(())
    if not bool(torch.isfinite(scalar).detach().cpu().item()):
        raise FloatingPointError(f"{name} must be finite")
    if lower is not None and bool((scalar < lower).detach().cpu().item()):
        raise ValueError(f"{name} must be at least {lower}")
    if upper is not None and bool((scalar > upper).detach().cpu().item()):
        raise ValueError(f"{name} must be at most {upper}")
    return scalar


def phase22_reference_norm(reference_core: torch.Tensor) -> torch.Tensor:
    """Return ``||Pi_R(B0)||_F`` without constructing a Phase21 cap."""

    reference = _require_float64_tensor(
        reference_core,
        name="reference_core",
        shape=(PHASE22_ROLE_COUNT, PHASE22_ROLE_COUNT),
    )
    right_gauged = reference - reference.mean(dim=-1, keepdim=True)
    norm = torch.linalg.vector_norm(right_gauged)
    if not bool((norm > 0).detach().cpu().item()):
        raise ValueError("the right-gauged parent core must have positive norm")
    return norm.detach()


def phase22_safety_bound(
    reference_core: torch.Tensor,
    *,
    ratio: float = PHASE22_SAFETY_RATIO,
) -> torch.Tensor:
    """Return the fail-closed coordinate radius; this is never a clip."""

    if not math.isfinite(float(ratio)) or float(ratio) <= 0:
        raise ValueError("safety ratio must be finite and positive")
    return (float(ratio) * phase22_reference_norm(reference_core)).detach()


def zero_phase22_state(
    *,
    device: torch.device | str | None = None,
) -> Phase22State:
    """Return the exact ``m=v=0`` state before the April 2010 replay."""

    position = torch.zeros(
        PHASE22_SUBSPACE_RANK,
        dtype=torch.float64,
        device=device,
    )
    return Phase22State(position=position, velocity=torch.zeros_like(position))


def validate_phase22_state(
    state: Phase22State,
    safety_bound: float | torch.Tensor,
    *,
    require_detached: bool = True,
) -> Phase22State:
    """Validate position and velocity independently, without rescaling either."""

    if not isinstance(state, Phase22State):
        raise TypeError("state must be a Phase22State")
    position = _require_float64_tensor(
        state.position,
        name="state.position",
        shape=(PHASE22_SUBSPACE_RANK,),
    )
    velocity = _require_float64_tensor(
        state.velocity,
        name="state.velocity",
        shape=(PHASE22_SUBSPACE_RANK,),
    )
    if position.device != velocity.device:
        raise ValueError("state position and velocity must use the same device")
    if require_detached and (position.requires_grad or velocity.requires_grad):
        raise ValueError("persistent state must be detached between episodes")
    bound = _float64_scalar(
        safety_bound,
        like=position,
        name="safety_bound",
        lower=0.0,
    )
    if not bool((bound > 0).detach().cpu().item()):
        raise ValueError("safety_bound must be strictly positive")
    if bool((torch.linalg.vector_norm(position) > bound).detach().cpu().item()):
        raise FloatingPointError("state position exceeds the Phase22 safety bound")
    if bool((torch.linalg.vector_norm(velocity) > bound).detach().cpu().item()):
        raise FloatingPointError("state velocity exceeds the Phase22 safety bound")
    return Phase22State(position=position, velocity=velocity)


def rotate_phase22_vector(
    value: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply ``x' = Qx`` to gradients, means, evidence, position, or velocity."""

    return rotate_coordinates(value, rotation)


def rotate_phase22_matrix(
    value: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply ``M' = Q M Q.T`` to K, covariance, or whitening matrices."""

    matrix = _require_float64_tensor(
        value,
        name="value",
        shape=(PHASE22_SUBSPACE_RANK, PHASE22_SUBSPACE_RANK),
    )
    validate_orthogonal_rotation(rotation)
    if matrix.device != rotation.device:
        raise ValueError("value and rotation must use the same device")
    return rotation @ matrix @ rotation.transpose(0, 1)


def rotate_phase22_state(
    state: Phase22State,
    rotation: torch.Tensor,
) -> Phase22State:
    """Rotate a detached persistent state into an equivalent basis frame."""

    if not isinstance(state, Phase22State):
        raise TypeError("state must be a Phase22State")
    if state.position.requires_grad or state.velocity.requires_grad:
        raise ValueError("persistent state must be detached before rotation")
    return Phase22State(
        position=rotate_phase22_vector(state.position, rotation).detach(),
        velocity=rotate_phase22_vector(state.velocity, rotation).detach(),
    )


def snapshot_phase22_state(state: Phase22State) -> dict[str, Any]:
    """Return a bit-preserving payload for exact snapshot/resume checks."""

    if not isinstance(state, Phase22State):
        raise TypeError("state must be a Phase22State")
    _require_float64_tensor(
        state.position,
        name="state.position",
        shape=(PHASE22_SUBSPACE_RANK,),
    )
    _require_float64_tensor(
        state.velocity,
        name="state.velocity",
        shape=(PHASE22_SUBSPACE_RANK,),
    )
    if state.position.requires_grad or state.velocity.requires_grad:
        raise ValueError("only detached state may be snapshotted")
    return {
        "schema_version": 1,
        "position": state.position.detach().clone(),
        "velocity": state.velocity.detach().clone(),
    }


def restore_phase22_state(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> Phase22State:
    """Restore a snapshot without arithmetic or dtype conversion."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if set(payload) != {"schema_version", "position", "velocity"}:
        raise ValueError("snapshot keys do not match the Phase22 schema")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported Phase22 snapshot schema")
    position = _require_float64_tensor(
        payload["position"],
        name="payload.position",
        shape=(PHASE22_SUBSPACE_RANK,),
    ).detach()
    velocity = _require_float64_tensor(
        payload["velocity"],
        name="payload.velocity",
        shape=(PHASE22_SUBSPACE_RANK,),
    ).detach()
    target = position.device if device is None else torch.device(device)
    return Phase22State(
        position=position.to(device=target).clone(),
        velocity=velocity.to(device=target).clone(),
    )


class PredictiveCrossLagObserver(nn.Module):
    """The 64-parameter Phase22 support-to-future-query observer."""

    def __init__(
        self,
        reference_core: torch.Tensor,
        basis: torch.Tensor,
        *,
        safety_ratio: float = PHASE22_SAFETY_RATIO,
    ) -> None:
        super().__init__()
        reference = _require_float64_tensor(
            reference_core,
            name="reference_core",
            shape=(PHASE22_ROLE_COUNT, PHASE22_ROLE_COUNT),
        )
        directions = _require_float64_tensor(
            basis,
            name="basis",
            shape=(
                PHASE22_SUBSPACE_RANK,
                PHASE22_ROLE_COUNT,
                PHASE22_ROLE_COUNT,
            ),
        )
        if reference.device != directions.device:
            raise ValueError("reference_core and basis must use the same device")
        validate_orthonormal_basis(directions)
        self.register_buffer("reference_core", reference.detach().clone())
        self.register_buffer("basis", directions.detach().clone())
        self.register_buffer(
            "reference_right_gauge_norm",
            phase22_reference_norm(reference),
        )
        self.register_buffer(
            "safety_bound",
            phase22_safety_bound(reference, ratio=safety_ratio),
        )
        self.K = nn.Parameter(
            torch.zeros(
                PHASE22_SUBSPACE_RANK,
                PHASE22_SUBSPACE_RANK,
                dtype=torch.float64,
                device=reference.device,
            )
        )

    def _validate_runtime(self) -> None:
        if self.K.dtype != torch.float64 or tuple(self.K.shape) != (
            PHASE22_SUBSPACE_RANK,
            PHASE22_SUBSPACE_RANK,
        ):
            raise TypeError("K must remain an 8-by-8 float64 parameter")
        if self.K.device != self.reference_core.device:
            raise ValueError("K and frozen Phase22 buffers must share a device")
        if not bool(torch.isfinite(self.K).all().detach().cpu().item()):
            raise FloatingPointError("K contains a non-finite value")
        if self.reference_core.dtype != torch.float64 or self.basis.dtype != torch.float64:
            raise TypeError("Phase22 frozen buffers must remain float64")

    def innovation(
        self,
        evidence: torch.Tensor,
        reliability: float | torch.Tensor,
    ) -> Phase22Innovation:
        """Return ``u=rho*K*z`` and its right-gauged operator displacement."""

        self._validate_runtime()
        z = _require_float64_tensor(
            evidence,
            name="evidence",
            shape=(PHASE22_SUBSPACE_RANK,),
        )
        if z.device != self.K.device:
            raise ValueError("evidence and K must use the same device")
        rho = _float64_scalar(
            reliability,
            like=z,
            name="reliability",
            lower=0.0,
            upper=1.0,
        )
        coordinates = rho * (self.K @ z)
        if not bool(torch.isfinite(coordinates).all().detach().cpu().item()):
            raise FloatingPointError("raw Phase22 innovation is non-finite")
        norm = torch.linalg.vector_norm(coordinates)
        if bool((norm > self.safety_bound).detach().cpu().item()):
            raise FloatingPointError("raw innovation exceeds the Phase22 safety bound")
        delta = operator_from_coordinates(coordinates, self.basis)
        return Phase22Innovation(
            evidence=z,
            reliability=rho,
            coordinates=coordinates,
            operator_delta=delta,
            coordinate_norm=norm,
        )

    @staticmethod
    def _detached_kl_scale(
        displacement: torch.Tensor,
        eta: float | torch.Tensor,
    ) -> torch.Tensor:
        value = _require_float64_tensor(displacement, name="displacement")
        scale = _float64_scalar(
            eta,
            like=value,
            name="eta",
            lower=0.0,
            upper=1.0,
        )
        if bool(torch.equal(value.detach(), torch.zeros_like(value))):
            return torch.ones((), dtype=torch.float64, device=value.device)
        return scale.detach()

    def static_update(
        self,
        evidence: torch.Tensor,
        reliability: float | torch.Tensor,
        eta: float | torch.Tensor,
    ) -> Phase22StaticOutput:
        """Build the reset/static live core while retaining gradients into K."""

        innovation = self.innovation(evidence, reliability)
        scale = self._detached_kl_scale(innovation.operator_delta, eta)
        realised_position = scale * innovation.coordinates
        if bool(
            (
                torch.linalg.vector_norm(realised_position) > self.safety_bound
            ).detach().cpu().item()
        ):
            raise FloatingPointError("realised position exceeds the Phase22 safety bound")
        realised_delta = scale * innovation.operator_delta
        core_float64 = self.reference_core + realised_delta
        live_core = core_float64.to(dtype=torch.float32)
        return Phase22StaticOutput(
            innovation=innovation,
            kl_scale=scale,
            realised_position=realised_position,
            realised_delta=realised_delta,
            core_float64=core_float64,
            live_core=live_core,
        )

    def propose_transition(
        self,
        state: Phase22State,
        evidence: torch.Tensor,
        reliability: float | torch.Tensor,
        *,
        beta: float | torch.Tensor = PHASE22_BETA,
        kappa: float | torch.Tensor = PHASE22_KAPPA,
    ) -> Phase22TransitionProposal:
        """Construct the uncapped M/V proposal before adjacent-state KL."""

        previous = validate_phase22_state(state, self.safety_bound)
        if previous.position.device != self.K.device:
            raise ValueError("state and observer must use the same device")
        innovation = self.innovation(evidence, reliability)
        beta_value = _float64_scalar(
            beta,
            like=previous.position,
            name="beta",
            lower=0.0,
            upper=1.0,
        )
        kappa_value = _float64_scalar(
            kappa,
            like=previous.position,
            name="kappa",
            lower=0.0,
            upper=1.0,
        )
        retained_velocity = beta_value * previous.velocity
        anchored_position = (1.0 - kappa_value) * previous.position
        proposed_position = (
            anchored_position + retained_velocity + innovation.coordinates
        )
        raw_displacement = proposed_position - previous.position
        tensors = (
            retained_velocity,
            anchored_position,
            proposed_position,
            raw_displacement,
        )
        if any(
            not bool(torch.isfinite(value).all().detach().cpu().item())
            for value in tensors
        ):
            raise FloatingPointError("Phase22 transition proposal is non-finite")
        return Phase22TransitionProposal(
            previous_state=previous,
            innovation=innovation,
            beta=beta_value,
            kappa=kappa_value,
            retained_velocity=retained_velocity,
            anchored_position=anchored_position,
            proposed_position=proposed_position,
            raw_displacement=raw_displacement,
        )

    def realize_transition(
        self,
        proposal: Phase22TransitionProposal,
        eta: float | torch.Tensor,
    ) -> Phase22TransitionOutput:
        """Apply detached adjacent-KL scale and store realised velocity exactly."""

        if not isinstance(proposal, Phase22TransitionProposal):
            raise TypeError("proposal must be a Phase22TransitionProposal")
        previous = validate_phase22_state(
            proposal.previous_state,
            self.safety_bound,
        )
        scale = self._detached_kl_scale(proposal.raw_displacement, eta)
        live_position = previous.position + scale * proposal.raw_displacement
        live_velocity = live_position - previous.position
        if not bool(torch.isfinite(live_position).all().detach().cpu().item()):
            raise FloatingPointError("realised position is non-finite")
        if not bool(torch.isfinite(live_velocity).all().detach().cpu().item()):
            raise FloatingPointError("realised velocity is non-finite")
        position_norm = torch.linalg.vector_norm(live_position)
        velocity_norm = torch.linalg.vector_norm(live_velocity)
        if bool((position_norm > self.safety_bound).detach().cpu().item()):
            raise FloatingPointError("realised position exceeds the Phase22 safety bound")
        if bool((velocity_norm > self.safety_bound).detach().cpu().item()):
            raise FloatingPointError("realised velocity exceeds the Phase22 safety bound")

        core_float64 = core_from_coordinates(
            self.reference_core,
            live_position,
            self.basis,
        )
        live_core = core_float64.to(dtype=torch.float32)
        state = Phase22State(
            position=live_position.detach().clone(),
            velocity=live_velocity.detach().clone(),
        )
        diagnostics = Phase22TransitionDiagnostics(
            beta=proposal.beta.detach().clone(),
            kappa=proposal.kappa.detach().clone(),
            safety_bound=self.safety_bound.detach().clone(),
            previous_position_norm=torch.linalg.vector_norm(
                previous.position
            ).detach(),
            previous_velocity_norm=torch.linalg.vector_norm(
                previous.velocity
            ).detach(),
            innovation_norm=proposal.innovation.coordinate_norm.detach(),
            retained_velocity_norm=torch.linalg.vector_norm(
                proposal.retained_velocity
            ).detach(),
            anchored_position_norm=torch.linalg.vector_norm(
                proposal.anchored_position
            ).detach(),
            proposed_position_norm=torch.linalg.vector_norm(
                proposal.proposed_position
            ).detach(),
            raw_displacement_norm=torch.linalg.vector_norm(
                proposal.raw_displacement
            ).detach(),
            kl_scale=scale.detach().clone(),
            realised_position_norm=position_norm.detach(),
            realised_velocity_norm=velocity_norm.detach(),
        )
        return Phase22TransitionOutput(
            state=state,
            live_position=live_position,
            live_velocity=live_velocity,
            core_float64=core_float64,
            live_core=live_core,
            diagnostics=diagnostics,
        )

    def rotated_copy(
        self,
        rotation: torch.Tensor,
    ) -> PredictiveCrossLagObserver:
        """Return the equivalent ``D'=QD, K'=QKQ.T`` observer frame."""

        validate_orthogonal_rotation(rotation)
        if rotation.device != self.K.device:
            raise ValueError("rotation and observer must use the same device")
        rotated = PredictiveCrossLagObserver(
            self.reference_core,
            rotate_basis(self.basis, rotation),
            safety_ratio=float(
                (self.safety_bound / self.reference_right_gauge_norm)
                .detach()
                .cpu()
            ),
        )
        with torch.no_grad():
            rotated.K.copy_(rotate_phase22_matrix(self.K.detach(), rotation))
        return rotated


__all__ = [
    "PHASE22_ALGEBRA_TOLERANCE",
    "PHASE22_BETA",
    "PHASE22_KAPPA",
    "PHASE22_ROLE_COUNT",
    "PHASE22_SAFETY_RATIO",
    "PHASE22_SUBSPACE_RANK",
    "Phase22Innovation",
    "Phase22State",
    "Phase22StaticOutput",
    "Phase22TransitionDiagnostics",
    "Phase22TransitionOutput",
    "Phase22TransitionProposal",
    "PredictiveCrossLagObserver",
    "phase22_reference_norm",
    "phase22_safety_bound",
    "restore_phase22_state",
    "rotate_phase22_matrix",
    "rotate_phase22_state",
    "rotate_phase22_vector",
    "snapshot_phase22_state",
    "validate_phase22_state",
    "zero_phase22_state",
]
