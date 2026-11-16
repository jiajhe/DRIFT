"""Zero-parameter state algebra for Phase21 subspace-posterior flow.

This module deliberately contains no model, parameter, optimiser, data
loading, or adjacency/KL implementation.  It only implements the locked
float64 algebra between an externally constructed orthonormal operator basis,
externally differentiated support observations, and an externally computed
adjacent-state KL bisection scale.

The separation is intentional.  A caller must first build a prior position,
differentiate the three mature support months at that position, solve one
pseudo-posterior correction, construct a capped proposal, and only then apply
the KL scale returned by the graph-side bisection.  The correction therefore
cannot accidentally be added twice.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch


PHASE21_ROLE_COUNT = 32
PHASE21_SUBSPACE_RANK = 8
PHASE21_SUPPORT_BLOCKS = 3
PHASE21_BETA = 2.0 ** (-1.0 / 12.0)
PHASE21_POSITION_RATIO = 0.20
PHASE21_VELOCITY_RATIO = 0.005
PHASE21_ALGEBRA_TOLERANCE = 1e-10
PHASE21_KL_HIT_TOLERANCE = 1e-7

PriorMode = Literal["candidate", "position_only"]
CorrectionMode = Literal["scalar_uncertainty", "constant_damping"]


@dataclass(frozen=True)
class SubspaceBasisDiagnostics:
    """Numerical invariants of the stored K32, rank-eight operator basis."""

    rank: int
    role_count: int
    right_gauge_max_abs_error: torch.Tensor
    gram_max_abs_error: torch.Tensor


@dataclass(frozen=True)
class OrthogonalRotationDiagnostics:
    """Numerical invariant of a coordinate-frame rotation."""

    rank: int
    gram_max_abs_error: torch.Tensor


@dataclass(frozen=True)
class Phase21Caps:
    """Locked coordinate radii derived from the right-gauged parent core."""

    reference_right_gauge_norm: torch.Tensor
    position_max: torch.Tensor
    velocity_max: torch.Tensor


@dataclass(frozen=True)
class Phase21State:
    """Detached persistent position and velocity in the rank-eight basis."""

    position: torch.Tensor
    velocity: torch.Tensor


@dataclass(frozen=True)
class Phase21Correction:
    """One scalar-uncertainty Gauss--Newton correction and its audit values."""

    mode: str
    mean_score: torch.Tensor
    curvature: torch.Tensor
    dispersion: torch.Tensor
    damping: torch.Tensor
    delta: torch.Tensor
    system: torch.Tensor
    correction: torch.Tensor
    curvature_min_eigenvalue: torch.Tensor
    curvature_spectral_norm: torch.Tensor


@dataclass(frozen=True)
class Phase21Prior:
    """State immediately before the current episode's support observation."""

    mode: str
    previous_state: Phase21State
    retained_velocity: torch.Tensor
    uncapped_position: torch.Tensor
    position: torch.Tensor
    position_max: torch.Tensor
    pre_position_hit: bool


@dataclass(frozen=True)
class Phase21TransitionProposal:
    """Velocity- and position-capped proposal before the graph-side KL guard."""

    prior: Phase21Prior
    correction: torch.Tensor
    raw_displacement: torch.Tensor
    capped_displacement: torch.Tensor
    uncapped_position: torch.Tensor
    position: torch.Tensor
    velocity_max: torch.Tensor
    velocity_hit: bool
    post_position_hit: bool

    @property
    def position_hit(self) -> bool:
        return self.prior.pre_position_hit or self.post_position_hit


@dataclass(frozen=True)
class Phase21TransitionDiagnostics:
    """Locked norms and hit definitions for one realised transition."""

    mode: str
    beta: torch.Tensor
    position_max: torch.Tensor
    velocity_max: torch.Tensor
    previous_position_norm: torch.Tensor
    previous_velocity_norm: torch.Tensor
    retained_velocity_norm: torch.Tensor
    pre_projection_position_norm: torch.Tensor
    prior_position_norm: torch.Tensor
    correction_norm: torch.Tensor
    raw_displacement_norm: torch.Tensor
    capped_displacement_norm: torch.Tensor
    post_projection_position_norm: torch.Tensor
    proposed_position_norm: torch.Tensor
    kl_scale: torch.Tensor
    realised_displacement_norm: torch.Tensor
    realised_position_norm: torch.Tensor
    stored_velocity_norm: torch.Tensor
    pre_position_hit: bool
    post_position_hit: bool
    position_hit: bool
    velocity_hit: bool
    kl_hit: bool


@dataclass(frozen=True)
class Phase21TransitionOutput:
    """Detached next state plus exhaustive state-side diagnostics."""

    state: Phase21State
    diagnostics: Phase21TransitionDiagnostics


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


def _as_float64_scalar(
    value: float | torch.Tensor,
    *,
    like: torch.Tensor,
    name: str,
    strictly_positive: bool = False,
    non_negative: bool = False,
) -> torch.Tensor:
    scalar = torch.as_tensor(value, dtype=torch.float64, device=like.device)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    scalar = scalar.reshape(())
    if not bool(torch.isfinite(scalar).detach().cpu().item()):
        raise FloatingPointError(f"{name} must be finite")
    if strictly_positive and not bool((scalar > 0).detach().cpu().item()):
        raise ValueError(f"{name} must be strictly positive")
    if non_negative and not bool((scalar >= 0).detach().cpu().item()):
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _hit_threshold(bound: torch.Tensor) -> torch.Tensor:
    """Return the exact preregistered absolute-plus-relative hit tolerance."""

    return bound + PHASE21_ALGEBRA_TOLERANCE * torch.maximum(
        torch.ones_like(bound), bound
    )


def _radial_cap(vector: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if bool((norm > maximum).detach().cpu().item()):
        if not bool((norm > 0).detach().cpu().item()):
            return torch.zeros_like(vector)
        return vector * (maximum / norm)
    return vector.clone()


def validate_orthonormal_basis(
    basis: torch.Tensor,
    *,
    tolerance: float = PHASE21_ALGEBRA_TOLERANCE,
) -> SubspaceBasisDiagnostics:
    """Validate the locked ``[8, 32, 32]`` right-gauged orthonormal basis.

    The sender-role axis is the final axis.  Only its mean is checked; no left
    projection or double centring is introduced here.
    """

    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("tolerance must be finite and positive")
    basis = _require_float64_tensor(
        basis,
        name="basis",
        shape=(PHASE21_SUBSPACE_RANK, PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
    )
    right_gauge_error = basis.mean(dim=-1).abs().amax()
    flattened = basis.reshape(PHASE21_SUBSPACE_RANK, -1)
    identity = torch.eye(
        PHASE21_SUBSPACE_RANK,
        dtype=torch.float64,
        device=basis.device,
    )
    gram_error = (flattened @ flattened.transpose(0, 1) - identity).abs().amax()
    if bool((right_gauge_error > float(tolerance)).detach().cpu().item()):
        raise ValueError(
            "basis violates the right-gauge tolerance: "
            f"{float(right_gauge_error.detach().cpu())} > {float(tolerance)}"
        )
    if bool((gram_error > float(tolerance)).detach().cpu().item()):
        raise ValueError(
            "basis violates the Frobenius-orthonormality tolerance: "
            f"{float(gram_error.detach().cpu())} > {float(tolerance)}"
        )
    return SubspaceBasisDiagnostics(
        rank=PHASE21_SUBSPACE_RANK,
        role_count=PHASE21_ROLE_COUNT,
        right_gauge_max_abs_error=right_gauge_error.detach(),
        gram_max_abs_error=gram_error.detach(),
    )


def validate_orthogonal_rotation(
    rotation: torch.Tensor,
    *,
    tolerance: float = PHASE21_ALGEBRA_TOLERANCE,
) -> OrthogonalRotationDiagnostics:
    """Validate an orthogonal change of coordinates in the locked subspace."""

    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("tolerance must be finite and positive")
    rotation = _require_float64_tensor(
        rotation,
        name="rotation",
        shape=(PHASE21_SUBSPACE_RANK, PHASE21_SUBSPACE_RANK),
    )
    identity = torch.eye(
        PHASE21_SUBSPACE_RANK,
        dtype=torch.float64,
        device=rotation.device,
    )
    error = (rotation @ rotation.transpose(0, 1) - identity).abs().amax()
    if bool((error > float(tolerance)).detach().cpu().item()):
        raise ValueError(
            "rotation violates orthogonality tolerance: "
            f"{float(error.detach().cpu())} > {float(tolerance)}"
        )
    return OrthogonalRotationDiagnostics(
        rank=PHASE21_SUBSPACE_RANK,
        gram_max_abs_error=error.detach(),
    )


def rotate_basis(
    basis: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Return ``D' = Q D`` for an equivalent orthonormal operator frame."""

    validate_orthonormal_basis(basis)
    validate_orthogonal_rotation(rotation)
    if basis.device != rotation.device:
        raise ValueError("basis and rotation must be on the same device")
    rotated = torch.einsum("ab,bij->aij", rotation, basis)
    validate_orthonormal_basis(rotated)
    return rotated


def rotate_coordinates(
    coordinates: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Return coordinates ``Q m`` matching :func:`rotate_basis`.

    Leading dimensions are preserved.  This helper is used for states,
    gradients, and corrections; it never assigns semantics to individual
    modes.
    """

    rotation = _require_float64_tensor(
        rotation,
        name="rotation",
        shape=(PHASE21_SUBSPACE_RANK, PHASE21_SUBSPACE_RANK),
    )
    validate_orthogonal_rotation(rotation)
    coordinates = _require_float64_tensor(coordinates, name="coordinates")
    if coordinates.ndim < 1 or coordinates.shape[-1] != PHASE21_SUBSPACE_RANK:
        raise ValueError("coordinates must end in the rank-eight mode axis")
    if coordinates.device != rotation.device:
        raise ValueError("coordinates and rotation must be on the same device")
    return coordinates @ rotation.transpose(0, 1)


def rotate_curvatures(
    curvatures: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Return ``Q C Q.T`` for one or more coordinate-space curvatures."""

    rotation = _require_float64_tensor(
        rotation,
        name="rotation",
        shape=(PHASE21_SUBSPACE_RANK, PHASE21_SUBSPACE_RANK),
    )
    validate_orthogonal_rotation(rotation)
    curvatures = _require_float64_tensor(curvatures, name="curvatures")
    if curvatures.ndim < 2 or tuple(curvatures.shape[-2:]) != (
        PHASE21_SUBSPACE_RANK,
        PHASE21_SUBSPACE_RANK,
    ):
        raise ValueError("curvatures must end in [8, 8]")
    if curvatures.device != rotation.device:
        raise ValueError("curvatures and rotation must be on the same device")
    return torch.einsum("ai,...ij,bj->...ab", rotation, curvatures, rotation)


def operator_from_coordinates(
    coordinates: torch.Tensor,
    basis: torch.Tensor,
    *,
    validate_basis: bool = True,
) -> torch.Tensor:
    """Map one rank-eight coordinate vector to its K32 operator displacement."""

    if validate_basis:
        validate_orthonormal_basis(basis)
    else:
        _require_float64_tensor(
            basis,
            name="basis",
            shape=(PHASE21_SUBSPACE_RANK, PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
        )
    coordinates = _require_float64_tensor(
        coordinates,
        name="coordinates",
        shape=(PHASE21_SUBSPACE_RANK,),
    )
    if coordinates.device != basis.device:
        raise ValueError("coordinates and basis must be on the same device")
    return torch.einsum("k,kij->ij", coordinates, basis)


def coordinates_from_operator(
    operator: torch.Tensor,
    basis: torch.Tensor,
    *,
    require_in_subspace: bool = True,
    tolerance: float = PHASE21_ALGEBRA_TOLERANCE,
) -> torch.Tensor:
    """Project a K32 displacement into coordinates, optionally checking loss."""

    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0:
        raise ValueError("tolerance must be finite and positive")
    validate_orthonormal_basis(basis)
    operator = _require_float64_tensor(
        operator,
        name="operator",
        shape=(PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
    )
    if operator.device != basis.device:
        raise ValueError("operator and basis must be on the same device")
    coordinates = torch.einsum("kij,ij->k", basis, operator)
    if require_in_subspace:
        reconstruction = operator_from_coordinates(
            coordinates,
            basis,
            validate_basis=False,
        )
        error = (operator - reconstruction).abs().amax()
        if bool((error > float(tolerance)).detach().cpu().item()):
            raise ValueError(
                "operator is not contained in the stored subspace: "
                f"max residual {float(error.detach().cpu())}"
            )
    return coordinates


def core_from_coordinates(
    reference_core: torch.Tensor,
    coordinates: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Construct the single live core ``B0 + sum_k m_k D_k``."""

    reference_core = _require_float64_tensor(
        reference_core,
        name="reference_core",
        shape=(PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
    )
    if reference_core.device != basis.device:
        raise ValueError("reference_core and basis must be on the same device")
    return reference_core + operator_from_coordinates(coordinates, basis)


def coordinates_from_core(
    core: torch.Tensor,
    reference_core: torch.Tensor,
    basis: torch.Tensor,
    *,
    require_in_subspace: bool = True,
    tolerance: float = PHASE21_ALGEBRA_TOLERANCE,
) -> torch.Tensor:
    """Recover coordinates of a core relative to its frozen parent ``B0``."""

    core = _require_float64_tensor(
        core,
        name="core",
        shape=(PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
    )
    reference_core = _require_float64_tensor(
        reference_core,
        name="reference_core",
        shape=(PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
    )
    if core.device != reference_core.device:
        raise ValueError("core and reference_core must be on the same device")
    return coordinates_from_operator(
        core - reference_core,
        basis,
        require_in_subspace=require_in_subspace,
        tolerance=tolerance,
    )


def phase21_caps(reference_core: torch.Tensor) -> Phase21Caps:
    """Derive ``Mmax`` and ``Vmax`` from ``||Pi_R(B0)||_F`` exactly once."""

    reference_core = _require_float64_tensor(
        reference_core,
        name="reference_core",
        shape=(PHASE21_ROLE_COUNT, PHASE21_ROLE_COUNT),
    )
    gauged = reference_core - reference_core.mean(dim=-1, keepdim=True)
    reference_norm = torch.linalg.vector_norm(gauged).detach()
    if not bool((reference_norm > 0).detach().cpu().item()):
        raise ValueError("the right-gauged parent core must have positive norm")
    return Phase21Caps(
        reference_right_gauge_norm=reference_norm,
        position_max=(PHASE21_POSITION_RATIO * reference_norm).detach(),
        velocity_max=(PHASE21_VELOCITY_RATIO * reference_norm).detach(),
    )


def zero_phase21_state(
    *,
    device: torch.device | str | None = None,
) -> Phase21State:
    """Return the locked zero state used before the April 2010 warm-up."""

    position = torch.zeros(
        PHASE21_SUBSPACE_RANK,
        dtype=torch.float64,
        device=device,
    )
    return Phase21State(position=position, velocity=torch.zeros_like(position))


def _validated_state(state: Phase21State, caps: Phase21Caps) -> Phase21State:
    if not isinstance(state, Phase21State):
        raise TypeError("state must be a Phase21State")
    position = _require_float64_tensor(
        state.position,
        name="state.position",
        shape=(PHASE21_SUBSPACE_RANK,),
    )
    velocity = _require_float64_tensor(
        state.velocity,
        name="state.velocity",
        shape=(PHASE21_SUBSPACE_RANK,),
    )
    if position.device != velocity.device:
        raise ValueError("state position and velocity must be on the same device")
    position_max = _as_float64_scalar(
        caps.position_max,
        like=position,
        name="caps.position_max",
        strictly_positive=True,
    )
    velocity_max = _as_float64_scalar(
        caps.velocity_max,
        like=position,
        name="caps.velocity_max",
        strictly_positive=True,
    )
    if bool(
        (
            torch.linalg.vector_norm(position) > _hit_threshold(position_max)
        ).detach().cpu().item()
    ):
        raise ValueError("incoming position exceeds the locked position cap")
    if bool(
        (
            torch.linalg.vector_norm(velocity) > _hit_threshold(velocity_max)
        ).detach().cpu().item()
    ):
        raise ValueError("incoming velocity exceeds the locked velocity cap")
    return Phase21State(
        position=position.detach().clone(),
        velocity=velocity.detach().clone(),
    )


def build_phase21_prior(
    state: Phase21State,
    caps: Phase21Caps,
    *,
    mode: PriorMode = "candidate",
) -> Phase21Prior:
    """Build the locked candidate or position-only pre-observation prior.

    The candidate uses ``v_minus = beta*v`` and radially projects
    ``m_prev+v_minus``.  The matched position-only control uses ``m_prev``
    exactly and requires its stored velocity to be zero.
    """

    if mode not in {"candidate", "position_only"}:
        raise ValueError(f"unsupported prior mode: {mode}")
    state = _validated_state(state, caps)
    position_max = _as_float64_scalar(
        caps.position_max,
        like=state.position,
        name="caps.position_max",
        strictly_positive=True,
    )
    if mode == "position_only":
        if not bool(torch.equal(state.velocity, torch.zeros_like(state.velocity))):
            raise ValueError("position_only state must store zero velocity")
        retained_velocity = torch.zeros_like(state.velocity)
        uncapped_position = state.position.clone()
        position = state.position.clone()
    else:
        retained_velocity = (PHASE21_BETA * state.velocity).detach()
        uncapped_position = (state.position + retained_velocity).detach()
        position = _radial_cap(uncapped_position, position_max).detach()
    pre_position_hit = bool(
        (
            torch.linalg.vector_norm(uncapped_position)
            > _hit_threshold(position_max)
        ).detach().cpu().item()
    )
    return Phase21Prior(
        mode=mode,
        previous_state=state,
        retained_velocity=retained_velocity,
        uncapped_position=uncapped_position,
        position=position,
        position_max=position_max.detach(),
        pre_position_hit=pre_position_hit,
    )


def solve_scalar_uncertainty_correction(
    debiased_block_scores: torch.Tensor,
    block_curvatures: torch.Tensor,
    *,
    lambda0: float | torch.Tensor,
    delta: float | torch.Tensor,
    s_train_squared: float | torch.Tensor,
    mode: CorrectionMode = "scalar_uncertainty",
) -> Phase21Correction:
    """Solve the locked three-block rank-eight pseudo-posterior correction.

    ``debiased_block_scores`` contains the three ``z_b`` vectors.  The common
    curvature is explicitly symmetrised, checked against the preregistered
    relative PSD tolerance, and solved only by Cholesky.  Inputs are detached
    so an online state can never retain a graph across episodes.
    """

    if mode not in {"scalar_uncertainty", "constant_damping"}:
        raise ValueError(f"unsupported correction mode: {mode}")
    scores = _require_float64_tensor(
        debiased_block_scores,
        name="debiased_block_scores",
        shape=(PHASE21_SUPPORT_BLOCKS, PHASE21_SUBSPACE_RANK),
    ).detach()
    curvatures = _require_float64_tensor(
        block_curvatures,
        name="block_curvatures",
        shape=(
            PHASE21_SUPPORT_BLOCKS,
            PHASE21_SUBSPACE_RANK,
            PHASE21_SUBSPACE_RANK,
        ),
    ).detach()
    if scores.device != curvatures.device:
        raise ValueError("scores and curvatures must be on the same device")
    lambda0_tensor = _as_float64_scalar(
        lambda0,
        like=scores,
        name="lambda0",
        strictly_positive=True,
    )
    delta_tensor = _as_float64_scalar(
        delta,
        like=scores,
        name="delta",
        strictly_positive=True,
    )
    s_train_tensor = _as_float64_scalar(
        s_train_squared,
        like=scores,
        name="s_train_squared",
        strictly_positive=True,
    )

    mean_score = scores.mean(dim=0)
    centered = scores - mean_score.unsqueeze(0)
    dispersion = centered.square().sum() / (
        PHASE21_SUBSPACE_RANK * (PHASE21_SUPPORT_BLOCKS - 1)
    )
    curvature = curvatures.mean(dim=0)
    curvature = 0.5 * (curvature + curvature.transpose(0, 1))
    if not bool(torch.isfinite(curvature).all().detach().cpu().item()):
        raise FloatingPointError("mean curvature is non-finite")

    eigenvalues = torch.linalg.eigvalsh(curvature)
    minimum = eigenvalues.amin()
    spectral_norm = eigenvalues.abs().amax()
    psd_floor = -PHASE21_ALGEBRA_TOLERANCE * torch.maximum(
        torch.ones_like(spectral_norm), spectral_norm
    )
    if bool((minimum < psd_floor).detach().cpu().item()):
        raise FloatingPointError(
            "curvature violates the locked PSD tolerance: "
            f"lambda_min={float(minimum.detach().cpu())}, "
            f"bound={float(psd_floor.detach().cpu())}"
        )

    if mode == "constant_damping":
        damping = 2.0 * lambda0_tensor
    else:
        damping = lambda0_tensor * (1.0 + dispersion / s_train_tensor)
    identity = torch.eye(
        PHASE21_SUBSPACE_RANK,
        dtype=torch.float64,
        device=scores.device,
    )
    system = curvature + (delta_tensor + damping) * identity
    system = 0.5 * (system + system.transpose(0, 1))
    cholesky, info = torch.linalg.cholesky_ex(system, check_errors=False)
    if int(info.detach().cpu().item()) != 0:
        raise FloatingPointError(
            "locked Cholesky solve failed; jitter and fallback solvers are forbidden"
        )
    correction = torch.cholesky_solve(
        -mean_score.unsqueeze(-1),
        cholesky,
    ).squeeze(-1)
    if not bool(torch.isfinite(correction).all().detach().cpu().item()):
        raise FloatingPointError("pseudo-posterior correction is non-finite")
    return Phase21Correction(
        mode=mode,
        mean_score=mean_score.detach(),
        curvature=curvature.detach(),
        dispersion=dispersion.detach(),
        damping=damping.detach(),
        delta=delta_tensor.detach(),
        system=system.detach(),
        correction=correction.detach(),
        curvature_min_eigenvalue=minimum.detach(),
        curvature_spectral_norm=spectral_norm.detach(),
    )


def normalized_gradient_correction(
    debiased_block_scores: torch.Tensor,
    *,
    amplitude: float | torch.Tensor,
) -> torch.Tensor:
    """Return the preregistered train-matched unit-gradient control step."""

    scores = _require_float64_tensor(
        debiased_block_scores,
        name="debiased_block_scores",
        shape=(PHASE21_SUPPORT_BLOCKS, PHASE21_SUBSPACE_RANK),
    ).detach()
    amplitude_tensor = _as_float64_scalar(
        amplitude,
        like=scores,
        name="amplitude",
        strictly_positive=True,
    )
    mean_score = scores.mean(dim=0)
    denominator = torch.linalg.vector_norm(mean_score).clamp_min(1e-12)
    return (-amplitude_tensor * mean_score / denominator).detach()


def propose_capped_transition(
    prior: Phase21Prior,
    correction: torch.Tensor,
    caps: Phase21Caps,
) -> Phase21TransitionProposal:
    """Add one correction, then cap displacement and position in that order."""

    if not isinstance(prior, Phase21Prior):
        raise TypeError("prior must be a Phase21Prior")
    correction = _require_float64_tensor(
        correction,
        name="correction",
        shape=(PHASE21_SUBSPACE_RANK,),
    ).detach()
    if correction.device != prior.position.device:
        raise ValueError("correction and prior must be on the same device")
    position_max = _as_float64_scalar(
        caps.position_max,
        like=correction,
        name="caps.position_max",
        strictly_positive=True,
    )
    velocity_max = _as_float64_scalar(
        caps.velocity_max,
        like=correction,
        name="caps.velocity_max",
        strictly_positive=True,
    )
    if not bool(torch.equal(position_max, prior.position_max)):
        raise ValueError("proposal caps do not match the caps used for its prior")

    previous_position = prior.previous_state.position
    raw_displacement = (prior.position + correction - previous_position).detach()
    raw_displacement_norm = torch.linalg.vector_norm(raw_displacement)
    velocity_hit = bool(
        (raw_displacement_norm > _hit_threshold(velocity_max)).detach().cpu().item()
    )
    capped_displacement = _radial_cap(raw_displacement, velocity_max).detach()
    uncapped_position = (previous_position + capped_displacement).detach()
    post_position_hit = bool(
        (
            torch.linalg.vector_norm(uncapped_position)
            > _hit_threshold(position_max)
        ).detach().cpu().item()
    )
    position = _radial_cap(uncapped_position, position_max).detach()
    return Phase21TransitionProposal(
        prior=prior,
        correction=correction,
        raw_displacement=raw_displacement,
        capped_displacement=capped_displacement,
        uncapped_position=uncapped_position,
        position=position,
        velocity_max=velocity_max.detach(),
        velocity_hit=velocity_hit,
        post_position_hit=post_position_hit,
    )


def realize_kl_scaled_transition(
    proposal: Phase21TransitionProposal,
    kl_scale: float | torch.Tensor,
) -> Phase21TransitionOutput:
    """Apply an external KL-feasible ``eta`` and store realised velocity.

    This function does not compute adjacency or KL.  It only realises a scale
    already chosen by the detached 24-step graph-side bisection along the
    segment from the previous position to the capped proposal.
    """

    if not isinstance(proposal, Phase21TransitionProposal):
        raise TypeError("proposal must be a Phase21TransitionProposal")
    previous_state = proposal.prior.previous_state
    eta = _as_float64_scalar(
        kl_scale,
        like=previous_state.position,
        name="kl_scale",
        non_negative=True,
    ).detach()
    if bool((eta > 1.0).detach().cpu().item()):
        raise ValueError("kl_scale must lie in [0, 1]")

    realised_position = (
        previous_state.position
        + eta * (proposal.position - previous_state.position)
    ).detach()
    realised_displacement = (realised_position - previous_state.position).detach()
    if proposal.prior.mode == "position_only":
        stored_velocity = torch.zeros_like(realised_displacement)
    else:
        stored_velocity = realised_displacement.clone()

    position_norm = torch.linalg.vector_norm(realised_position)
    displacement_norm = torch.linalg.vector_norm(realised_displacement)
    velocity_norm = torch.linalg.vector_norm(stored_velocity)
    if bool(
        (
            position_norm > _hit_threshold(proposal.prior.position_max)
        ).detach().cpu().item()
    ):
        raise FloatingPointError("KL-scaled position escaped the position cap")
    if bool(
        (
            displacement_norm > _hit_threshold(proposal.velocity_max)
        ).detach().cpu().item()
    ):
        raise FloatingPointError("KL-scaled displacement escaped the velocity cap")

    diagnostics = Phase21TransitionDiagnostics(
        mode=proposal.prior.mode,
        beta=previous_state.position.new_tensor(PHASE21_BETA),
        position_max=proposal.prior.position_max,
        velocity_max=proposal.velocity_max,
        previous_position_norm=torch.linalg.vector_norm(previous_state.position),
        previous_velocity_norm=torch.linalg.vector_norm(previous_state.velocity),
        retained_velocity_norm=torch.linalg.vector_norm(
            proposal.prior.retained_velocity
        ),
        pre_projection_position_norm=torch.linalg.vector_norm(
            proposal.prior.uncapped_position
        ),
        prior_position_norm=torch.linalg.vector_norm(proposal.prior.position),
        correction_norm=torch.linalg.vector_norm(proposal.correction),
        raw_displacement_norm=torch.linalg.vector_norm(proposal.raw_displacement),
        capped_displacement_norm=torch.linalg.vector_norm(
            proposal.capped_displacement
        ),
        post_projection_position_norm=torch.linalg.vector_norm(
            proposal.uncapped_position
        ),
        proposed_position_norm=torch.linalg.vector_norm(proposal.position),
        kl_scale=eta,
        realised_displacement_norm=displacement_norm,
        realised_position_norm=position_norm,
        stored_velocity_norm=velocity_norm,
        pre_position_hit=proposal.prior.pre_position_hit,
        post_position_hit=proposal.post_position_hit,
        position_hit=proposal.position_hit,
        velocity_hit=proposal.velocity_hit,
        kl_hit=bool(
            (eta < 1.0 - PHASE21_KL_HIT_TOLERANCE).detach().cpu().item()
        ),
    )
    return Phase21TransitionOutput(
        state=Phase21State(
            position=realised_position,
            velocity=stored_velocity.detach(),
        ),
        diagnostics=diagnostics,
    )
