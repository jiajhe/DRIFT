from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from .directed_graph import role_scores, sender_softmax


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def _logit(value: float) -> float:
    return log(value / (1.0 - value))


def _atanh(value: float) -> float:
    return 0.5 * log((1.0 + value) / (1.0 - value))


class _Entmax15Function(torch.autograd.Function):
    """Exact Entmax1.5 forward with its support-safe analytical Jacobian.

    The threshold calculation follows the sorting algorithm used by the
    DeepSPIN reference implementation.  Backpropagating through the threshold
    square root directly is unsafe at a changing sparse-support boundary, so
    backward uses the closed-form Jacobian-vector product instead.
    """

    @staticmethod
    def forward(ctx: object, logits: torch.Tensor, dim: int) -> torch.Tensor:
        # Entmax is translation invariant.  Removing the maximum prevents the
        # threshold moments from losing precision as role logits sharpen.
        shifted = logits - logits.max(dim=dim, keepdim=True).values
        scaled = shifted / 2.0
        sorted_logits, _ = torch.sort(scaled, dim=dim, descending=True)
        dimension = logits.shape[dim]
        view = [1] * logits.ndim
        view[dim] = dimension
        ranks = torch.arange(
            1,
            dimension + 1,
            dtype=logits.dtype,
            device=logits.device,
        ).reshape(view)

        cumulative = sorted_logits.cumsum(dim)
        cumulative_sq = sorted_logits.square().cumsum(dim)
        mean = cumulative / ranks
        mean_sq = cumulative_sq / ranks
        variance_sum = ranks * (mean_sq - mean.square())
        delta = (1.0 - variance_sum) / ranks
        taus = mean - torch.sqrt(torch.clamp_min(delta, 0.0))
        support = taus <= sorted_logits
        support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau = taus.gather(dim, support_size - 1)
        probabilities = torch.clamp_min(scaled - tau, 0.0).square()

        ctx.dim = dim  # type: ignore[attr-defined]
        ctx.save_for_backward(probabilities)  # type: ignore[attr-defined]
        return probabilities

    @staticmethod
    def backward(
        ctx: object, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (probabilities,) = ctx.saved_tensors  # type: ignore[attr-defined]
        dim = ctx.dim  # type: ignore[attr-defined]
        support_slope = probabilities.sqrt()
        grad_logits = grad_output * support_slope
        correction = grad_logits.sum(dim=dim, keepdim=True) / support_slope.sum(
            dim=dim, keepdim=True
        )
        grad_logits = grad_logits - correction * support_slope
        return grad_logits, None


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse :math:`\alpha=1.5` entmax with stable analytical backward."""

    if logits.ndim == 0:
        raise ValueError("entmax15 expects at least one dimension")
    if logits.shape[dim] <= 0:
        raise ValueError("entmax15 cannot normalize an empty dimension")
    if not logits.is_floating_point():
        raise TypeError("entmax15 expects floating-point logits")
    _require_finite("entmax15 logits", logits)

    probabilities = _Entmax15Function.apply(logits, dim % logits.ndim)
    _require_finite("entmax15 probabilities", probabilities)
    return probabilities


def center_cross_section(
    values: torch.Tensor,
    stock_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-center the final stock axis and zero padded stocks."""

    if values.ndim < 1:
        raise ValueError("values must have a stock axis")
    _require_finite("cross-sectional values", values)
    if stock_mask is None:
        return values - values.mean(dim=-1, keepdim=True)
    if stock_mask.shape != values.shape:
        raise ValueError("stock_mask must match the cross-sectional value shape")
    if stock_mask.dtype != torch.bool:
        raise TypeError("stock_mask must be boolean")
    count = stock_mask.sum(dim=-1, keepdim=True)
    if torch.any(count == 0):
        raise ValueError("each cross section must contain at least one valid stock")
    weight = stock_mask.to(dtype=values.dtype)
    mean = (values * weight).sum(dim=-1, keepdim=True) / count.to(values.dtype)
    return (values - mean) * weight


def receiver_absolute_normalize(
    receiver_roles: torch.Tensor,
    channel_signal: torch.Tensor,
    absolute_strength: torch.Tensor,
    *,
    eps: float = 1e-6,
    stock_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distribute signed channel signals with receiver absolute normalization.

    ``receiver_roles[..., i, k]`` is receiver stock ``i``'s membership in
    latent market role ``k``.  The denominator uses non-negative absolute
    channel strengths, so signs are controlled solely by ``channel_signal``.
    """

    if receiver_roles.ndim < 2:
        raise ValueError("receiver_roles must end in [stock, role]")
    if channel_signal.shape != receiver_roles.shape[:-2] + receiver_roles.shape[-1:]:
        raise ValueError("channel_signal must match receiver leading and role axes")
    if absolute_strength.shape != channel_signal.shape:
        try:
            absolute_strength = torch.broadcast_to(absolute_strength, channel_signal.shape)
        except RuntimeError as error:
            raise ValueError("absolute_strength must broadcast to channel_signal") from error
    if eps <= 0:
        raise ValueError("eps must be positive")
    for name, value in (
        ("receiver_roles", receiver_roles),
        ("channel_signal", channel_signal),
        ("absolute_strength", absolute_strength),
    ):
        _require_finite(name, value)
    if torch.any(receiver_roles < 0):
        raise ValueError("receiver_roles must be non-negative")
    if torch.any(absolute_strength < 0):
        raise ValueError("absolute_strength must be non-negative")

    denominator = torch.einsum(
        "...nk,...k->...n", receiver_roles, absolute_strength
    ).add(float(eps))
    raw = torch.einsum("...nk,...k->...n", receiver_roles, channel_signal)
    raw = raw / denominator
    if stock_mask is not None:
        if stock_mask.shape != raw.shape or stock_mask.dtype != torch.bool:
            raise ValueError("boolean stock_mask must match receiver stock axes")
        raw = raw.masked_fill(~stock_mask, 0.0)
        denominator = denominator.masked_fill(~stock_mask, 1.0)
    return raw, denominator


def _double_center_operator(
    operator: torch.Tensor,
    stock_mask: torch.Tensor | None,
) -> torch.Tensor:
    if operator.ndim < 2 or operator.shape[-1] != operator.shape[-2]:
        raise ValueError("operator must end in square [receiver, sender] axes")
    _require_finite("transport operator", operator)
    n = operator.shape[-1]
    if stock_mask is None:
        row_mean = operator.mean(dim=-1, keepdim=True)
        column_mean = operator.mean(dim=-2, keepdim=True)
        grand_mean = operator.mean(dim=(-2, -1), keepdim=True)
        return operator - row_mean - column_mean + grand_mean
    expected = operator.shape[:-1]
    if stock_mask.shape != expected or stock_mask.dtype != torch.bool:
        raise ValueError("boolean stock_mask must match operator receiver axes")
    count = stock_mask.sum(dim=-1, keepdim=True)
    if torch.any(count == 0):
        raise ValueError("each operator must contain at least one valid stock")
    valid = stock_mask.to(dtype=operator.dtype)
    pair = valid.unsqueeze(-1) * valid.unsqueeze(-2)
    masked = operator * pair
    divisor = count.to(dtype=operator.dtype)
    row_mean = masked.sum(dim=-1, keepdim=True) / divisor.unsqueeze(-1)
    column_mean = masked.sum(dim=-2, keepdim=True) / divisor.unsqueeze(-1)
    grand_mean = masked.sum(dim=(-2, -1), keepdim=True) / divisor.square().unsqueeze(-1)
    centered = masked - row_mean * pair - column_mean * pair + grand_mean * pair
    return centered * pair


def signed_transport_operator(
    sender_roles: torch.Tensor,
    receiver_roles: torch.Tensor,
    signed_strength: torch.Tensor,
    absolute_strength: torch.Tensor,
    *,
    eps: float = 1e-6,
    stock_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build ``T[..., i, j]`` for sender ``j`` -> receiver ``i`` flow.

    Sender memberships are normalized over the sender-stock set.  Receiver
    memberships are normalized by total absolute transmission strength, then
    both stock axes are centered, implementing ``C Rbar D Sbar.T C``.
    """

    if sender_roles.shape != receiver_roles.shape or sender_roles.ndim < 2:
        raise ValueError("sender and receiver roles must share [stock, role] shape")
    if eps <= 0:
        raise ValueError("eps must be positive")
    role_shape = sender_roles.shape[:-2] + sender_roles.shape[-1:]
    for name, strength in (
        ("signed_strength", signed_strength),
        ("absolute_strength", absolute_strength),
    ):
        try:
            strength = torch.broadcast_to(strength, role_shape)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to latent-role axes") from error
        if name == "signed_strength":
            signed_strength = strength
        else:
            absolute_strength = strength
    for name, value in (
        ("sender_roles", sender_roles),
        ("receiver_roles", receiver_roles),
        ("signed_strength", signed_strength),
        ("absolute_strength", absolute_strength),
    ):
        _require_finite(name, value)
    if torch.any(sender_roles < 0) or torch.any(receiver_roles < 0):
        raise ValueError("latent market role memberships must be non-negative")
    if torch.any(absolute_strength < 0):
        raise ValueError("absolute_strength must be non-negative")

    if stock_mask is not None:
        if stock_mask.shape != sender_roles.shape[:-1] or stock_mask.dtype != torch.bool:
            raise ValueError("boolean stock_mask must match role stock axes")
        valid = stock_mask.unsqueeze(-1).to(dtype=sender_roles.dtype)
        sender_roles = sender_roles * valid
        receiver_roles = receiver_roles * valid
    sender_denominator = sender_roles.sum(dim=-2, keepdim=True).add(float(eps))
    sender_normalized = sender_roles / sender_denominator
    receiver_denominator = torch.einsum(
        "...nk,...k->...n", receiver_roles, absolute_strength
    ).add(float(eps))
    receiver_normalized = receiver_roles / receiver_denominator.unsqueeze(-1)
    raw = torch.einsum(
        "...ik,...k,...jk->...ij",
        receiver_normalized,
        signed_strength,
        sender_normalized,
    )
    return _double_center_operator(raw, stock_mask)


def lag_convolve(
    current: torch.Tensor,
    history: torch.Tensor,
    lag_weights: torch.Tensor,
    lags: Sequence[int],
) -> torch.Tensor:
    """Apply per-channel discrete lag kernels.

    ``history[..., k, d-1]`` stores channel ``k``'s unfiltered carrier from
    ``d`` trading days ago.  Lag zero reads ``current``.
    """

    lags = tuple(int(lag) for lag in lags)
    if not lags or lags[0] != 0 or any(lag < 0 for lag in lags):
        raise ValueError("lags must be non-negative and start with zero")
    if len(set(lags)) != len(lags) or tuple(sorted(lags)) != lags:
        raise ValueError("lags must be unique and increasing")
    if current.ndim < 1:
        raise ValueError("current must have a latent-role axis")
    max_lag = max(lags)
    if history.shape != current.shape + (max_lag,):
        raise ValueError(
            "history must append max(lags) past-carrier slots to current shape"
        )
    expected_weights = current.shape[-1:] + (len(lags),)
    if lag_weights.shape != expected_weights:
        try:
            lag_weights = torch.broadcast_to(
                lag_weights, current.shape[:-1] + expected_weights
            )
        except RuntimeError as error:
            raise ValueError("lag_weights must match [role, configured_lag]") from error
    _require_finite("current carrier", current)
    _require_finite("carrier history", history)
    _require_finite("lag weights", lag_weights)

    selected = []
    for lag in lags:
        selected.append(current if lag == 0 else history[..., lag - 1])
    carriers = torch.stack(selected, dim=-1)
    return (carriers * lag_weights).sum(dim=-1)


def update_carrier_history(current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Prepend today's unfiltered channel carrier to the lag state."""

    if history.shape[:-1] != current.shape:
        raise ValueError("history must append a lag axis to current")
    if history.shape[-1] == 0:
        return history
    return torch.cat([current.unsqueeze(-1), history[..., :-1]], dim=-1)


@dataclass(frozen=True)
class StatefulFlowState:
    """Chronological flow state aligned to the current stock axis.

    ``v_prev`` and ``p_local_prev`` must already be mapped by instrument to the
    next day's stock axis.  Stock identity alignment and zeroing re-entries are
    deliberately trainer responsibilities.  ``carrier_history`` is shared
    latent-channel history and does not have a stock axis.
    """

    v_prev: torch.Tensor
    p_local_prev: torch.Tensor
    carrier_history: torch.Tensor

    def detach(self) -> StatefulFlowState:
        """Detach state tensors for trainer-controlled truncated BPTT."""

        return StatefulFlowState(
            v_prev=self.v_prev.detach(),
            p_local_prev=self.p_local_prev.detach(),
            carrier_history=self.carrier_history.detach(),
        )


@dataclass(frozen=True)
class StatefulFlowOutput:
    prediction: torch.Tensor
    p_local: torch.Tensor
    v: torch.Tensor
    state: StatefulFlowState
    diagnostics: Mapping[str, torch.Tensor]


class StatefulFlowFDG(nn.Module):
    """End-to-end Stateful Flow-FDG with persistent signed information flow.

    The primary input is one chronological day's causal feature windows with
    shape ``[stock, lookback, feature]``.  Optional leading batch dimensions
    are supported, but recurrence still advances one day per call.

    The signed flow diagnostic is named ``T``.  ``A`` is reserved for the
    probability diagnostic ``softmax_sender(R @ B @ S.T)`` and therefore obeys
    the repository-wide sender ``j`` -> receiver ``i`` convention.
    """

    SOURCE_MODES = {"level", "level_innovation"}

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 64,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        local_hidden_dim: int = 64,
        n_roles: int = 16,
        role_dim: int | None = None,
        gate_feature_dim: int = 16,
        gate_hidden_dim: int = 32,
        context_dim: int = 0,
        lags: Sequence[int] = (0, 1, 2, 3, 5),
        source_mode: str = "level",
        dropout: float = 0.1,
        encoder_dropout: float | None = None,
        local_dropout: float | None = None,
        tau_sender: float = 1.0,
        tau_receiver: float = 1.0,
        rho_init: float = 0.5,
        rho_max: float = 0.9,
        signed_strength_init: float = 0.02,
        reliability_init: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("d_model", d_model),
            ("gru_hidden_dim", gru_hidden_dim),
            ("gru_num_layers", gru_num_layers),
            ("local_hidden_dim", local_hidden_dim),
            ("n_roles", n_roles),
            ("gate_feature_dim", gate_feature_dim),
            ("gate_hidden_dim", gate_hidden_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(context_dim) < 0:
            raise ValueError("context_dim must be non-negative")
        resolved_encoder_dropout = (
            float(dropout) if encoder_dropout is None else float(encoder_dropout)
        )
        resolved_local_dropout = (
            float(dropout) if local_dropout is None else float(local_dropout)
        )
        if not 0.0 <= resolved_encoder_dropout < 1.0:
            raise ValueError("encoder_dropout must lie in [0, 1)")
        if not 0.0 <= resolved_local_dropout < 1.0:
            raise ValueError("local_dropout must lie in [0, 1)")
        if min(float(tau_sender), float(tau_receiver)) <= 0:
            raise ValueError("role temperatures must be positive")
        if not 0.0 < float(rho_max) < 1.0:
            raise ValueError("rho_max must lie strictly between zero and one")
        if not 0.0 < float(rho_init) < float(rho_max):
            raise ValueError("rho_init must lie strictly between zero and rho_max")
        if not 0.0 < abs(float(signed_strength_init)) < 1.0:
            raise ValueError("signed_strength_init magnitude must lie in (0, 1)")
        if not 0.0 < float(reliability_init) < 1.0:
            raise ValueError("reliability_init must lie strictly between zero and one")
        if float(eps) <= 0:
            raise ValueError("eps must be positive")
        if source_mode not in self.SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {sorted(self.SOURCE_MODES)}")

        parsed_lags = tuple(int(lag) for lag in lags)
        if (
            not parsed_lags
            or parsed_lags[0] != 0
            or tuple(sorted(set(parsed_lags))) != parsed_lags
        ):
            raise ValueError("lags must be unique, increasing, and start with zero")

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.gru_num_layers = int(gru_num_layers)
        self.local_hidden_dim = int(local_hidden_dim)
        self.n_roles = int(n_roles)
        self.role_dim = int(role_dim if role_dim is not None else d_model)
        if self.role_dim <= 0:
            raise ValueError("role_dim must be positive")
        self.gate_feature_dim = int(gate_feature_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.context_dim = int(context_dim)
        self.lags = parsed_lags
        self.max_lag = max(parsed_lags)
        self.source_mode = str(source_mode)
        self.dropout = float(dropout)
        self.encoder_dropout = resolved_encoder_dropout
        self.local_dropout = resolved_local_dropout
        self.tau_sender = float(tau_sender)
        self.tau_receiver = float(tau_receiver)
        self.rho_max = float(rho_max)
        self.eps = float(eps)

        self.temporal_encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=self.gru_num_layers,
            batch_first=True,
            dropout=self.encoder_dropout if self.gru_num_layers > 1 else 0.0,
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.local_head = nn.Sequential(
            nn.Linear(self.d_model, self.local_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.local_dropout),
            nn.Linear(self.local_hidden_dim, 1),
        )

        self.role_norm = nn.LayerNorm(self.d_model)
        self.sender_query = nn.Linear(self.d_model, self.role_dim, bias=False)
        self.receiver_query = nn.Linear(self.d_model, self.role_dim, bias=False)
        self.sender_prototypes = nn.Parameter(torch.empty(self.n_roles, self.role_dim))
        self.receiver_prototypes = nn.Parameter(torch.empty(self.n_roles, self.role_dim))

        self.source_gate = nn.Linear(self.d_model, 1)
        self.set_value = nn.Linear(self.d_model, self.gate_feature_dim, bias=False)
        gate_input_dim = 2 * self.gate_feature_dim + 2 + self.context_dim
        self.set_gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, 1),
        )

        sign = torch.where(
            torch.arange(self.n_roles) % 2 == 0,
            torch.ones(self.n_roles),
            -torch.ones(self.n_roles),
        )
        raw_strength = _atanh(abs(float(signed_strength_init)))
        self.alpha = nn.Parameter(sign * raw_strength)
        self.gamma = nn.Parameter(
            torch.full((self.n_roles,), _logit(float(reliability_init)))
        )
        self.beta = nn.Parameter(torch.zeros(self.n_roles, len(self.lags)))
        self.rho_logit = nn.Parameter(
            torch.tensor(_logit(float(rho_init) / self.rho_max))
        )
        self.eta = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize role prototypes without touching grammar priors."""

        nn.init.orthogonal_(self.sender_prototypes)
        nn.init.orthogonal_(self.receiver_prototypes)

    @property
    def rho(self) -> torch.Tensor:
        return self.rho_max * torch.sigmoid(self.rho_logit)

    @property
    def flow_lambda(self) -> torch.Tensor:
        return torch.tanh(self.eta)

    def structure_dict(self) -> dict[str, object]:
        """Return a JSON-serializable architecture snapshot for run logs."""

        return {
            "model": type(self).__name__,
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "temporal_encoder": "gru",
            "gru_hidden_dim": self.gru_hidden_dim,
            "gru_num_layers": self.gru_num_layers,
            "local_hidden_dim": self.local_hidden_dim,
            "n_roles": self.n_roles,
            "role_dim": self.role_dim,
            "role_normalizer": "entmax15",
            "role_normalizer_forward": "max_shifted_exact_sorting_threshold",
            "role_normalizer_backward": "closed_form_support_safe_jacobian_vector_product",
            "gate_feature_dim": self.gate_feature_dim,
            "gate_hidden_dim": self.gate_hidden_dim,
            "context_dim": self.context_dim,
            "lags": list(self.lags),
            "source_mode": self.source_mode,
            "dropout": self.dropout,
            "encoder_dropout": self.encoder_dropout,
            "local_dropout": self.local_dropout,
            "tau_sender": self.tau_sender,
            "tau_receiver": self.tau_receiver,
            "rho_parameterization": "rho_max*sigmoid",
            "rho_max": self.rho_max,
            "lambda_parameterization": "tanh_eta",
            "eta_init": 0.0,
            "transport_name": "T",
            "adjacency_diagnostic": "A=softmax_sender(R@B@S.T)",
        }

    def parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Separate local and flow parameters for staged optimization."""

        local_modules: Iterable[nn.Module] = (
            self.temporal_encoder,
            self.temporal_projection,
            self.local_head,
        )
        local = tuple(
            parameter
            for module in local_modules
            for parameter in module.parameters()
        )
        local_ids = {id(parameter) for parameter in local}
        flow = tuple(
            parameter for parameter in self.parameters() if id(parameter) not in local_ids
        )
        return {"local": local, "flow": flow}

    def _validate_features(
        self,
        features: torch.Tensor,
        stock_mask: torch.Tensor | None,
    ) -> tuple[tuple[int, ...], int, int]:
        if features.ndim < 3:
            raise ValueError("features must end in [stock, lookback, feature]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"feature dimension {features.shape[-1]} != configured {self.input_dim}"
            )
        if features.shape[-2] <= 0 or features.shape[-3] <= 0:
            raise ValueError("stock and lookback axes must be non-empty")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        _require_finite("features", features)
        leading = tuple(features.shape[:-3])
        stocks = int(features.shape[-3])
        lookback = int(features.shape[-2])
        if stock_mask is not None:
            if stock_mask.shape != leading + (stocks,):
                raise ValueError("stock_mask must match feature leading and stock axes")
            if stock_mask.dtype != torch.bool:
                raise TypeError("stock_mask must be boolean")
            if torch.any(stock_mask.sum(dim=-1) == 0):
                raise ValueError("each day must contain at least one valid stock")
        return leading, stocks, lookback

    def _encode(self, features: torch.Tensor) -> torch.Tensor:
        leading = features.shape[:-3]
        stocks, lookback = features.shape[-3:-1]
        packed = features.reshape(-1, lookback, self.input_dim)
        _, hidden = self.temporal_encoder(packed)
        z = self.temporal_projection(hidden[-1])
        return z.reshape(*leading, stocks, self.d_model)

    def _initial_state(
        self,
        z: torch.Tensor,
    ) -> StatefulFlowState:
        leading = z.shape[:-2]
        stocks = z.shape[-2]
        return StatefulFlowState(
            v_prev=z.new_zeros(*leading, stocks),
            p_local_prev=z.new_zeros(*leading, stocks),
            carrier_history=z.new_zeros(*leading, self.n_roles, self.max_lag),
        )

    def _validate_state(
        self,
        state: StatefulFlowState,
        *,
        leading: tuple[int, ...],
        stocks: int,
        reference: torch.Tensor,
    ) -> None:
        expected_stock = leading + (stocks,)
        expected_history = leading + (self.n_roles, self.max_lag)
        if state.v_prev.shape != expected_stock:
            raise ValueError(
                "state.v_prev must already be instrument-aligned to current stock axes"
            )
        if state.p_local_prev.shape != expected_stock:
            raise ValueError(
                "state.p_local_prev must already be instrument-aligned to current stock axes"
            )
        if state.carrier_history.shape != expected_history:
            raise ValueError("state.carrier_history does not match configured roles/lags")
        for name, value in (
            ("state.v_prev", state.v_prev),
            ("state.p_local_prev", state.p_local_prev),
            ("state.carrier_history", state.carrier_history),
        ):
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must share feature dtype and device")
            _require_finite(name, value)

    def _role_memberships(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.role_norm(z)
        scale = sqrt(float(self.role_dim))
        sender_logits = torch.einsum(
            "...nd,kd->...nk", self.sender_query(normalized), self.sender_prototypes
        ) / (scale * self.tau_sender)
        receiver_logits = torch.einsum(
            "...nd,kd->...nk",
            self.receiver_query(normalized),
            self.receiver_prototypes,
        ) / (scale * self.tau_receiver)
        return entmax15(sender_logits, dim=-1), entmax15(receiver_logits, dim=-1)

    def _set_statistics(
        self,
        z: torch.Tensor,
        sender_normalized: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        values = self.set_value(z)
        mean = torch.einsum("...nk,...nd->...kd", sender_normalized, values)
        centered = values.unsqueeze(-2) - mean.unsqueeze(-3)
        variance = torch.einsum(
            "...nk,...nkd->...kd", sender_normalized, centered.square()
        )
        second_moment = torch.einsum(
            "...nk,...nd->...kd", sender_normalized, values.square()
        )
        consistency = mean.square().sum(dim=-1, keepdim=True) / (
            second_moment.sum(dim=-1, keepdim=True) + self.eps
        )
        effective_count = sender_normalized.square().sum(dim=-2).add(self.eps).reciprocal()
        gate_parts = [
            mean,
            torch.log1p(variance),
            consistency,
            torch.log1p(effective_count).unsqueeze(-1),
        ]
        leading = z.shape[:-2]
        if self.context_dim:
            if context is None or context.shape != leading + (self.context_dim,):
                raise ValueError("context must match feature leading axes and context_dim")
            _require_finite("context", context)
            gate_parts.append(context.unsqueeze(-2).expand(*leading, self.n_roles, -1))
        elif context is not None:
            raise ValueError("context was provided but context_dim is zero")
        gate_input = torch.cat(gate_parts, dim=-1)
        gate = torch.sigmoid(self.set_gate(gate_input).squeeze(-1))
        return gate, {
            "set_mean": mean,
            "set_variance": variance,
            "set_consistency": consistency.squeeze(-1),
            "set_effective_count": effective_count,
        }

    def forward(
        self,
        features: torch.Tensor,
        *,
        state: StatefulFlowState | None = None,
        stock_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        compute_graph_diagnostics: bool = True,
        compute_transport_diagnostics: bool = False,
    ) -> StatefulFlowOutput:
        leading, stocks, _ = self._validate_features(features, stock_mask)
        z = self._encode(features)
        p_local = self.local_head(z).squeeze(-1)
        if stock_mask is not None:
            p_local = p_local.masked_fill(~stock_mask, 0.0)
        active_state = self._initial_state(z) if state is None else state
        self._validate_state(
            active_state,
            leading=leading,
            stocks=stocks,
            reference=z,
        )

        sender_roles, receiver_roles = self._role_memberships(z)
        if stock_mask is not None:
            valid = stock_mask.unsqueeze(-1).to(dtype=z.dtype)
            sender_roles = sender_roles * valid
            receiver_roles = receiver_roles * valid
        sender_denominator = sender_roles.sum(dim=-2, keepdim=True).add(self.eps)
        sender_normalized = sender_roles / sender_denominator

        omega = torch.sigmoid(self.source_gate(z).squeeze(-1))
        if self.source_mode == "level":
            source_raw = p_local
        else:
            innovation = p_local - active_state.p_local_prev
            source_raw = omega * p_local + (1.0 - omega) * innovation
        u = center_cross_section(source_raw, stock_mask)
        payload = u + self.rho * active_state.v_prev
        if stock_mask is not None:
            payload = payload.masked_fill(~stock_mask, 0.0)

        carrier = torch.einsum("...nk,...n->...k", sender_normalized, payload)
        lag_weights = torch.softmax(self.beta, dim=-1)
        delayed_carrier = lag_convolve(
            carrier,
            active_state.carrier_history,
            lag_weights,
            self.lags,
        )
        gate, set_diagnostics = self._set_statistics(z, sender_normalized, context)
        signed_direction = torch.tanh(self.alpha)
        reliability = torch.sigmoid(self.gamma)
        signed_strength = reliability * signed_direction * gate
        absolute_strength = reliability * signed_direction.abs() * gate
        channel_signal = signed_strength * delayed_carrier
        v_raw, receiver_denominator = receiver_absolute_normalize(
            receiver_roles,
            channel_signal,
            absolute_strength,
            eps=self.eps,
            stock_mask=stock_mask,
        )
        v = center_cross_section(v_raw, stock_mask)
        prediction = p_local + self.flow_lambda * v
        if stock_mask is not None:
            prediction = prediction.masked_fill(~stock_mask, 0.0)

        next_state = StatefulFlowState(
            v_prev=v,
            p_local_prev=p_local,
            carrier_history=update_carrier_history(
                carrier, active_state.carrier_history
            ),
        )
        diagnostics: dict[str, torch.Tensor] = {
            "z": z,
            "S": sender_roles,
            "R": receiver_roles,
            "S_normalized": sender_normalized,
            "a": signed_direction,
            "r": reliability,
            "pi": lag_weights,
            "g": gate,
            "omega": omega,
            "u": u,
            "b": payload,
            "carrier": carrier,
            "delayed_carrier": delayed_carrier,
            "channel_signal": channel_signal,
            "receiver_abs_denominator": receiver_denominator,
            "v_raw": v_raw,
            "rho": self.rho,
            "lambda": self.flow_lambda,
            **set_diagnostics,
        }
        if compute_graph_diagnostics or compute_transport_diagnostics:
            T = signed_transport_operator(
                sender_roles,
                receiver_roles,
                signed_strength,
                absolute_strength,
                eps=self.eps,
                stock_mask=stock_mask,
            )
            diagnostics["T"] = T
        if compute_graph_diagnostics:
            B = torch.diag(reliability * signed_direction)
            scores = role_scores(receiver_roles, B, sender_roles)
            A = sender_softmax(scores, tau_a=1.0, stock_mask=stock_mask)
            diagnostics.update({"B": B, "scores": scores, "A": A})
        for name, value in (
            ("p_local", p_local),
            ("flow state", v),
            ("prediction", prediction),
            *diagnostics.items(),
        ):
            _require_finite(name, value)
        return StatefulFlowOutput(
            prediction=prediction,
            p_local=p_local,
            v=v,
            state=next_state,
            diagnostics=diagnostics,
        )


__all__ = [
    "StatefulFlowFDG",
    "StatefulFlowOutput",
    "StatefulFlowState",
    "center_cross_section",
    "entmax15",
    "lag_convolve",
    "receiver_absolute_normalize",
    "signed_transport_operator",
    "update_carrier_history",
]
