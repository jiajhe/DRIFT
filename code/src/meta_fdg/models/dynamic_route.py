from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .regressor import StaticFDGRegressor


def _logit(value: float) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError("logit input must lie strictly inside (0, 1)")
    return float(math.log(float(value) / (1.0 - float(value))))


def _atanh(value: float) -> float:
    if not -1.0 < float(value) < 1.0:
        raise ValueError("atanh input must lie strictly inside (-1, 1)")
    return float(math.atanh(float(value)))


def center_and_bound_role_bias(raw: torch.Tensor, maximum: float) -> torch.Tensor:
    """Centre the role axis and enforce a per-vector absolute logit cap."""

    if raw.ndim < 1:
        raise ValueError("a role bias must have a role axis")
    if float(maximum) <= 0:
        raise ValueError("maximum role bias must be positive")
    value = float(maximum) * torch.tanh(raw)
    value = value - value.mean(dim=-1, keepdim=True)
    absolute_max = value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(maximum) / absolute_max, max=1.0)
    return value * scale


def cyclic_shift_role_bias(value: torch.Tensor, shift: int) -> torch.Tensor:
    """Cyclically shift only the latent-role axis of a complete correction.

    Positive one maps role ``r`` to role ``(r + 1) mod K``.  Sender and
    receiver corrections call this function independently: the former changes
    the sender-role columns of ``R @ B @ S.T`` and the latter changes its
    receiver-role rows.
    """

    if value.ndim < 1 or int(value.shape[-1]) <= 0:
        raise ValueError("a route correction must have a non-empty role axis")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("role shift must be an integer")
    if shift == 0:
        return value
    return torch.roll(value, shifts=int(shift), dims=-1)


@dataclass(frozen=True)
class RouteProposal:
    sender: torch.Tensor
    receiver: torch.Tensor


@dataclass(frozen=True)
class RouteEpisodeControls:
    sender_gate: torch.Tensor
    receiver_gate: torch.Tensor
    sender_prior: torch.Tensor
    receiver_prior: torch.Tensor
    alarm: torch.Tensor
    sender_kl_budget: torch.Tensor
    receiver_kl_budget: torch.Tensor


class CausalRouteProposal(nn.Module):
    """A role-only recurrent proposal over an independently processed stream."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_roles: int,
        *,
        lookback: int = 60,
        architecture: str = "lstm",
        max_logit_bias: float = 0.15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(int(input_dim), int(hidden_dim), int(n_roles), int(lookback)) <= 0:
            raise ValueError("route proposal dimensions must be positive")
        if architecture not in {"lstm", "gru"}:
            raise ValueError("route architecture must be lstm or gru")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_roles = int(n_roles)
        self.lookback = int(lookback)
        self.architecture = str(architecture)
        self.max_logit_bias = float(max_logit_bias)
        self.dropout = nn.Dropout(float(dropout))
        recurrent = nn.LSTM if architecture == "lstm" else nn.GRU
        self.temporal = recurrent(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.sender_head = nn.Linear(self.hidden_dim, self.n_roles)
        self.receiver_head = nn.Linear(self.hidden_dim, self.n_roles)
        # Epoch -1 must be an exact Static-FDG identity even though the
        # recurrent encoder itself has ordinary random initialization.
        nn.init.zeros_(self.sender_head.weight)
        nn.init.zeros_(self.sender_head.bias)
        nn.init.zeros_(self.receiver_head.weight)
        nn.init.zeros_(self.receiver_head.bias)

    def forward(
        self,
        route_history: torch.Tensor,
        route_lengths: torch.Tensor,
    ) -> RouteProposal:
        if route_history.ndim < 3:
            raise ValueError("route history must end in [lookback, feature]")
        if route_history.shape[-2:] != (self.lookback, self.input_dim):
            raise ValueError("route history shape differs from proposal configuration")
        if route_lengths.shape != route_history.shape[:-2]:
            raise ValueError("route lengths must match history leading dimensions")
        flat_history = route_history.reshape(-1, self.lookback, self.input_dim)
        flat_lengths = route_lengths.reshape(-1).to(dtype=torch.long)
        if torch.any(flat_lengths < 0) or torch.any(flat_lengths > self.lookback):
            raise ValueError("route lengths must lie in [0, lookback]")
        # Zero-length entries are padded stocks.  Packing a single zero vector
        # keeps the recurrent call valid; the downstream stock mask removes it.
        packed = nn.utils.rnn.pack_padded_sequence(
            flat_history,
            flat_lengths.clamp_min(1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.temporal(packed)
        if self.architecture == "lstm":
            hidden = hidden[0]
        encoded = self.dropout(self.norm(hidden[-1]))
        leading = route_history.shape[:-2]
        sender = self.sender_head(encoded).reshape(*leading, self.n_roles)
        receiver = self.receiver_head(encoded).reshape(*leading, self.n_roles)
        return RouteProposal(
            sender=center_and_bound_role_bias(sender, self.max_logit_bias),
            receiver=center_and_bound_role_bias(receiver, self.max_logit_bias),
        )


class CurrentOnlyRouteProposal(nn.Module):
    """Near-parameter-matched control that reads only the current feature row."""

    def __init__(
        self,
        input_dim: int,
        n_roles: int,
        *,
        lookback: int = 60,
        first_hidden_dim: int = 64,
        second_hidden_dim: int = 58,
        max_logit_bias: float = 0.15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(
            int(input_dim),
            int(n_roles),
            int(lookback),
            int(first_hidden_dim),
            int(second_hidden_dim),
        ) <= 0:
            raise ValueError("route proposal dimensions must be positive")
        self.input_dim = int(input_dim)
        self.n_roles = int(n_roles)
        self.lookback = int(lookback)
        self.max_logit_bias = float(max_logit_bias)
        self.current = nn.Sequential(
            nn.Linear(self.input_dim, int(first_hidden_dim)),
            nn.LayerNorm(int(first_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(first_hidden_dim), int(second_hidden_dim)),
            nn.LayerNorm(int(second_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.sender_head = nn.Linear(int(second_hidden_dim), self.n_roles)
        self.receiver_head = nn.Linear(int(second_hidden_dim), self.n_roles)
        nn.init.zeros_(self.sender_head.weight)
        nn.init.zeros_(self.sender_head.bias)
        nn.init.zeros_(self.receiver_head.weight)
        nn.init.zeros_(self.receiver_head.bias)

    def forward(
        self,
        route_history: torch.Tensor,
        route_lengths: torch.Tensor,
    ) -> RouteProposal:
        if route_history.ndim < 3:
            raise ValueError("route history must end in [lookback, feature]")
        if route_history.shape[-2:] != (self.lookback, self.input_dim):
            raise ValueError("route history shape differs from proposal configuration")
        if route_lengths.shape != route_history.shape[:-2]:
            raise ValueError("route lengths must match history leading dimensions")
        flat = route_history.reshape(-1, self.lookback, self.input_dim)
        lengths = route_lengths.reshape(-1).to(dtype=torch.long)
        if torch.any(lengths < 0) or torch.any(lengths > self.lookback):
            raise ValueError("route lengths must lie in [0, lookback]")
        rows = torch.arange(flat.shape[0], device=flat.device)
        current = flat[rows, (lengths - 1).clamp_min(0)]
        encoded = self.current(current)
        leading = route_history.shape[:-2]
        sender = self.sender_head(encoded).reshape(*leading, self.n_roles)
        receiver = self.receiver_head(encoded).reshape(*leading, self.n_roles)
        return RouteProposal(
            sender=center_and_bound_role_bias(sender, self.max_logit_bias),
            receiver=center_and_bound_role_bias(receiver, self.max_logit_bias),
        )


class RouteSupportController(nn.Module):
    """Map detached mature-support role statistics to market-level controls."""

    def __init__(
        self,
        n_roles: int,
        *,
        n_horizons: int = 1,
        n_summaries: int = 2,
        n_statistics: int = 4,
        hidden_dim: int = 16,
        prior_logit_cap: float = 0.05,
        gate_init: float = 0.1,
        kl_epsilon_min: float = 2e-4,
        kl_epsilon_max: float = 2e-3,
    ) -> None:
        super().__init__()
        if min(
            int(n_roles),
            int(n_horizons),
            int(n_summaries),
            int(n_statistics),
            int(hidden_dim),
        ) <= 0:
            raise ValueError("route controller dimensions must be positive")
        if not 0 <= float(kl_epsilon_min) <= float(kl_epsilon_max) or float(
            kl_epsilon_max
        ) <= 0:
            raise ValueError("route KL budgets must satisfy 0 <= min <= max and max > 0")
        self.n_roles = int(n_roles)
        self.n_horizons = int(n_horizons)
        self.n_summaries = int(n_summaries)
        self.n_statistics = int(n_statistics)
        self.hidden_dim = int(hidden_dim)
        self.prior_logit_cap = float(prior_logit_cap)
        self.kl_epsilon_min = float(kl_epsilon_min)
        self.kl_epsilon_max = float(kl_epsilon_max)
        role_input_dim = self.n_horizons * self.n_summaries * self.n_statistics
        self.role_encoder = nn.Sequential(
            nn.Linear(role_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.prior_head = nn.Linear(2 * self.hidden_dim, 2)
        self.control_head = nn.Linear(self.hidden_dim, 2)
        nn.init.zeros_(self.prior_head.weight)
        nn.init.zeros_(self.prior_head.bias)
        nn.init.zeros_(self.control_head.weight)
        with torch.no_grad():
            self.control_head.bias.copy_(
                torch.tensor([_logit(gate_init), _logit(gate_init)])
            )

    def forward(self, support_role_state: torch.Tensor) -> RouteEpisodeControls:
        expected = (
            self.n_horizons,
            self.n_summaries,
            self.n_statistics,
            self.n_roles,
        )
        if support_role_state.shape != expected:
            raise ValueError(
                f"support role state must have shape {expected}, got "
                f"{tuple(support_role_state.shape)}"
            )
        # The current query loss may train the controller, but it must never
        # backpropagate into the support pass or frozen Stage-I model.
        state = support_role_state.detach()
        role_input = state.permute(3, 0, 1, 2).reshape(self.n_roles, -1)
        encoded = self.role_encoder(role_input)
        pooled = torch.cat(
            [encoded.mean(dim=0), encoded.std(dim=0, unbiased=False)], dim=0
        )
        global_context = self.global_encoder(pooled)
        repeated_context = global_context.unsqueeze(0).expand(self.n_roles, -1)
        prior_raw = self.prior_head(torch.cat([encoded, repeated_context], dim=-1))
        sender_prior = center_and_bound_role_bias(
            prior_raw[:, 0], self.prior_logit_cap
        )
        receiver_prior = center_and_bound_role_bias(
            prior_raw[:, 1], self.prior_logit_cap
        )
        controls = self.control_head(global_context)
        sender_gate, receiver_gate = torch.sigmoid(controls).unbind(dim=-1)
        # The support-derived trust gate also sets the available route budget;
        # this avoids a separate non-identifiable alarm head in v1.
        alarm = 0.5 * (sender_gate + receiver_gate)
        sender_epsilon = self.kl_epsilon_min + sender_gate * (
            self.kl_epsilon_max - self.kl_epsilon_min
        )
        receiver_epsilon = self.kl_epsilon_min + receiver_gate * (
            self.kl_epsilon_max - self.kl_epsilon_min
        )
        return RouteEpisodeControls(
            sender_gate=sender_gate,
            receiver_gate=receiver_gate,
            sender_prior=sender_prior,
            receiver_prior=receiver_prior,
            alarm=alarm,
            sender_kl_budget=sender_epsilon,
            receiver_kl_budget=receiver_epsilon,
        )


class ProbeRouteSupportController(nn.Module):
    """Gate a route using detached full-probe evidence from mature support."""

    def __init__(
        self,
        n_roles: int,
        *,
        global_feature_dim: int = 16,
        role_feature_dim: int = 2,
        hidden_dim: int = 16,
        prior_logit_cap: float = 0.05,
        gate_init: float = 0.5,
        kl_epsilon_min: float = 2e-4,
        kl_epsilon_max: float = 2e-3,
        signed_gate: bool = False,
        gain_coherence_normalization: bool = False,
    ) -> None:
        super().__init__()
        if min(
            int(n_roles),
            int(global_feature_dim),
            int(role_feature_dim),
            int(hidden_dim),
        ) <= 0:
            raise ValueError("probe controller dimensions must be positive")
        if not 0 <= float(kl_epsilon_min) <= float(kl_epsilon_max) or float(
            kl_epsilon_max
        ) <= 0:
            raise ValueError("route KL budgets must satisfy 0 <= min <= max and max > 0")
        self.n_roles = int(n_roles)
        self.global_feature_dim = int(global_feature_dim)
        self.role_feature_dim = int(role_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.prior_logit_cap = float(prior_logit_cap)
        self.kl_epsilon_min = float(kl_epsilon_min)
        self.kl_epsilon_max = float(kl_epsilon_max)
        self.signed_gate = bool(signed_gate)
        self.gain_coherence_normalization = bool(gain_coherence_normalization)
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.sender_role_encoder = nn.Sequential(
            nn.Linear(self.role_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.receiver_role_encoder = nn.Sequential(
            nn.Linear(self.role_feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.prior_head = nn.ModuleDict(
            {
                "sender": nn.Linear(2 * self.hidden_dim, 1),
                "receiver": nn.Linear(2 * self.hidden_dim, 1),
            }
        )
        self.control_head = nn.Linear(self.hidden_dim, 2)
        for head in self.prior_head.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.control_head.weight)
        initial_gate_bias = (
            _atanh(gate_init) if self.signed_gate else _logit(gate_init)
        )
        with torch.no_grad():
            self.control_head.bias.copy_(
                torch.tensor([initial_gate_bias, initial_gate_bias])
            )

    @property
    def state_dim(self) -> int:
        return self.global_feature_dim + 2 * self.n_roles * self.role_feature_dim

    def forward(self, probe_state: torch.Tensor) -> RouteEpisodeControls:
        if probe_state.shape != (self.state_dim,):
            raise ValueError(
                f"probe route state must have shape {(self.state_dim,)}, got "
                f"{tuple(probe_state.shape)}"
            )
        state = probe_state.detach()
        global_features = state[: self.global_feature_dim]
        sender_start = self.global_feature_dim
        receiver_start = sender_start + self.n_roles * self.role_feature_dim
        sender_role_features = state[sender_start:receiver_start].reshape(
            self.n_roles, self.role_feature_dim
        )
        receiver_role_features = state[receiver_start:].reshape(
            self.n_roles, self.role_feature_dim
        )
        global_encoded = self.global_encoder(global_features)
        sender_role_encoded = self.sender_role_encoder(sender_role_features)
        receiver_role_encoded = self.receiver_role_encoder(receiver_role_features)
        context = self.context_encoder(
            torch.cat(
                [
                    global_encoded,
                    sender_role_encoded.mean(dim=0),
                    sender_role_encoded.std(dim=0, unbiased=False),
                    receiver_role_encoded.mean(dim=0),
                    receiver_role_encoded.std(dim=0, unbiased=False),
                ],
                dim=0,
            )
        )
        repeated_context = context.unsqueeze(0).expand(self.n_roles, -1)
        sender_prior_raw = self.prior_head["sender"](
            torch.cat([sender_role_encoded, repeated_context], dim=-1)
        ).squeeze(-1)
        receiver_prior_raw = self.prior_head["receiver"](
            torch.cat([receiver_role_encoded, repeated_context], dim=-1)
        ).squeeze(-1)
        sender_prior = center_and_bound_role_bias(
            sender_prior_raw, self.prior_logit_cap
        )
        receiver_prior = center_and_bound_role_bias(
            receiver_prior_raw, self.prior_logit_cap
        )
        control_logits = self.control_head(context)
        gates = (
            torch.tanh(control_logits)
            if self.signed_gate
            else torch.sigmoid(control_logits)
        )
        sender_gate, receiver_gate = gates.unbind(dim=-1)
        sender_magnitude = sender_gate.abs() if self.signed_gate else sender_gate
        receiver_magnitude = (
            receiver_gate.abs() if self.signed_gate else receiver_gate
        )
        alarm = 0.5 * (sender_magnitude + receiver_magnitude)
        width = self.kl_epsilon_max - self.kl_epsilon_min
        return RouteEpisodeControls(
            sender_gate=sender_gate,
            receiver_gate=receiver_gate,
            sender_prior=sender_prior,
            receiver_prior=receiver_prior,
            alarm=alarm,
            sender_kl_budget=self.kl_epsilon_min + sender_magnitude * width,
            receiver_kl_budget=self.kl_epsilon_min + receiver_magnitude * width,
        )


class GatedDynamicRoleRegressor(nn.Module):
    """A frozen Static-FDG with a strictly role-only dynamic route."""

    def __init__(
        self,
        base: StaticFDGRegressor,
        proposal: CausalRouteProposal | CurrentOnlyRouteProposal,
        controller: RouteSupportController | ProbeRouteSupportController,
    ) -> None:
        super().__init__()
        if int(base.temporal_lookback) != 1:
            raise ValueError("dynamic route requires an ordinary current-day Stage-I base")
        if base.graph.role_mode != "separate":
            raise ValueError("Stage-II dynamic route requires separate sender/receiver roles")
        n_roles = int(base.graph.B0.shape[-1])
        if proposal.n_roles != n_roles or controller.n_roles != n_roles:
            raise ValueError("base, proposal, and controller role dimensions must match")
        self.base = base
        self.proposal = proposal
        self.controller = controller
        self.base.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True) -> GatedDynamicRoleRegressor:
        super().train(mode)
        # Frozen dropout/batch behaviour must be identical to Stage-I replay.
        self.base.eval()
        self.base.requires_grad_(False)
        # In the second stage of warmup-then-controller training, the proposal
        # is a fixed causal operator.  Its query path must be deterministic and
        # match the eval-mode proposal used to build the support probe.
        if not any(parameter.requires_grad for parameter in self.proposal.parameters()):
            self.proposal.eval()
        return self

    def episode_controls(
        self, support_role_state: torch.Tensor
    ) -> RouteEpisodeControls:
        return self.controller(support_role_state)

    def complete_route_correction(
        self,
        proposal: RouteProposal,
        controls: RouteEpisodeControls,
    ) -> RouteProposal:
        """Apply the episode controller to both complete role corrections.

        The returned values are centred, unprojected logit corrections.  A
        role-shift control must act here, before the graph independently
        projects each sender/receiver correction into its own KL trust region.
        """

        if proposal.sender.shape != proposal.receiver.shape:
            raise ValueError("sender and receiver proposals must have matching shapes")
        n_roles = int(self.base.graph.B0.shape[-1])
        if proposal.sender.ndim < 1 or proposal.sender.shape[-1] != n_roles:
            raise ValueError("route proposal has an incompatible latent-role axis")
        for name, prior in (
            ("sender", controls.sender_prior),
            ("receiver", controls.receiver_prior),
        ):
            if prior.ndim != 1 or prior.shape[0] != n_roles:
                raise ValueError(f"{name} route prior must have shape [role]")
        if (
            isinstance(self.controller, ProbeRouteSupportController)
            and self.controller.signed_gate
        ):
            # A signed meta-gain controls the complete episode correction.
            # Consequently g=0 is a true abstention and g<0 reverses both the
            # temporal proposal and its support-conditioned market prior.
            sender_bias = controls.sender_gate * (
                proposal.sender + controls.sender_prior
            )
            receiver_bias = controls.receiver_gate * (
                proposal.receiver + controls.receiver_prior
            )
        else:
            # Preserve the exact checkpoint semantics of all unsigned route
            # experiments, where the market prior was an independent term.
            sender_bias = (
                controls.sender_gate * proposal.sender + controls.sender_prior
            )
            receiver_bias = (
                controls.receiver_gate * proposal.receiver + controls.receiver_prior
            )
        return RouteProposal(
            sender=sender_bias - sender_bias.mean(dim=-1, keepdim=True),
            receiver=receiver_bias - receiver_bias.mean(dim=-1, keepdim=True),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        route_history: torch.Tensor | None = None,
        route_lengths: torch.Tensor | None = None,
        route_proposal: RouteProposal | None = None,
        controls: RouteEpisodeControls | None = None,
        stock_mask: torch.Tensor | None = None,
        core: torch.Tensor | None = None,
        route_enabled: bool = True,
        sender_role_shift: int = 0,
        receiver_role_shift: int = 0,
        apply_kl_budget: bool = True,
        return_graph: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not route_enabled:
            if route_proposal is not None or sender_role_shift or receiver_role_shift:
                raise ValueError("route-off cannot accept a proposal or role shift")
            return self.base(
                features,
                stock_mask=stock_mask,
                core=core,
                return_graph=return_graph,
            )
        if controls is None:
            raise ValueError("dynamic route requires episode controls")
        if route_proposal is None:
            if route_history is None or route_lengths is None:
                raise ValueError("dynamic route requires history and lengths")
            proposal = self.proposal(route_history, route_lengths)
        else:
            if route_history is not None or route_lengths is not None:
                raise ValueError(
                    "a precomputed route proposal cannot be mixed with history inputs"
                )
            proposal = route_proposal
        expected_proposal_shape = (*features.shape[:-1], int(self.base.B0.shape[-1]))
        if (
            proposal.sender.shape != expected_proposal_shape
            or proposal.receiver.shape != expected_proposal_shape
        ):
            raise ValueError(
                "route proposal must match feature leading axes and latent roles"
            )
        correction = self.complete_route_correction(proposal, controls)
        sender_bias = cyclic_shift_role_bias(correction.sender, sender_role_shift)
        receiver_bias = cyclic_shift_role_bias(
            correction.receiver, receiver_role_shift
        )

        # Only the graph role logits may see the temporal stream.  The frozen
        # representation and skip path remain functions of current FDG input.
        with torch.no_grad():
            representation = self.base.encoder(features)
        graph_output = self.base.graph(
            representation,
            core=core,
            stock_mask=stock_mask,
            sender_role_bias=sender_bias,
            receiver_role_bias=receiver_bias,
            sender_role_kl_budget=(
                controls.sender_kl_budget if apply_kl_budget else None
            ),
            receiver_role_kl_budget=(
                controls.receiver_kl_budget if apply_kl_budget else None
            ),
        )
        fused = self.base.skip_projection(representation) + graph_output["message"]
        embedding = self.base.hidden(fused)
        prediction = self.base.output(embedding).squeeze(-1)
        if stock_mask is not None:
            prediction = prediction.masked_fill(~stock_mask, 0.0)
        if not return_graph:
            return prediction
        return {
            "prediction": prediction,
            "embedding": embedding,
            "representation": representation,
            "route_sender_proposal": proposal.sender,
            "route_receiver_proposal": proposal.receiver,
            "route_sender_bias": sender_bias,
            "route_receiver_bias": receiver_bias,
            "route_sender_gate": controls.sender_gate,
            "route_receiver_gate": controls.receiver_gate,
            "route_alarm": controls.alarm,
            **graph_output,
        }


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
