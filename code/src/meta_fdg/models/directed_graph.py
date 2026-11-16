from __future__ import annotations

import torch
from torch import nn


def role_scores(receiver_roles: torch.Tensor, core: torch.Tensor, sender_roles: torch.Tensor) -> torch.Tensor:
    """Return scores whose ``[..., i, j]`` entry is sender ``j`` -> receiver ``i``.

    ``core[..., a, b]`` means sender role ``b`` -> receiver role ``a``.
    Receiver and sender role tensors end in ``[stock, role]`` and may have
    arbitrary matching batch dimensions.
    """

    if receiver_roles.shape[-1] != core.shape[-2]:
        raise ValueError("receiver role dimension must match core receiver axis")
    if sender_roles.shape[-1] != core.shape[-1]:
        raise ValueError("sender role dimension must match core sender axis")
    return receiver_roles @ core @ sender_roles.transpose(-1, -2)


def sender_softmax(
    scores: torch.Tensor,
    tau_a: float | torch.Tensor,
    stock_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize over sender-stock columns, never over receiver rows."""

    temperature = torch.as_tensor(tau_a, dtype=scores.dtype, device=scores.device)
    if torch.any(temperature <= 0):
        raise ValueError("tau_a must be positive")
    logits = scores / temperature
    if stock_mask is not None:
        if stock_mask.shape != scores.shape[:-1]:
            raise ValueError("stock_mask must match score receiver dimensions")
        logits = logits.masked_fill(~stock_mask.unsqueeze(-2), -torch.inf)
    adjacency = torch.softmax(logits, dim=-1)
    adjacency = torch.nan_to_num(adjacency, nan=0.0)
    if stock_mask is not None:
        adjacency = adjacency * stock_mask.unsqueeze(-1).to(adjacency.dtype)
    return adjacency


def _broadcast_role_bias(
    bias: torch.Tensor | None,
    *,
    logits: torch.Tensor,
) -> torch.Tensor | None:
    """Centre a role-logit bias and broadcast it over the stock axis."""

    if bias is None:
        return None
    if bias.shape[-1] != logits.shape[-1]:
        raise ValueError("role bias must match the graph role dimension")
    if bias.ndim > logits.ndim:
        raise ValueError("role bias has too many dimensions for graph logits")
    value = bias.to(dtype=logits.dtype, device=logits.device)
    value = value - value.mean(dim=-1, keepdim=True)
    while value.ndim < logits.ndim:
        value = value.unsqueeze(-2)
    return value


def _normalise_daily_budget(
    budget: float | torch.Tensor,
    *,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Return a non-negative KL budget for every independent query day."""

    daily_shape = logits.shape[:-2]
    value = torch.as_tensor(budget, dtype=logits.dtype, device=logits.device)
    if torch.any(value < 0):
        raise ValueError("role KL budget must be non-negative")
    if not daily_shape:
        if value.numel() != 1:
            raise ValueError("unbatched role logits require a scalar KL budget")
        return value.reshape(())
    try:
        return torch.broadcast_to(value, daily_shape)
    except RuntimeError as error:
        raise ValueError(
            f"role KL budget must broadcast to daily shape {daily_shape}"
        ) from error


def _masked_daily_role_kl(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    stock_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean ``KL(candidate || reference)`` over stocks, separately per day."""

    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference roles must have matching shapes")
    per_stock = (
        candidate
        * (
            candidate.clamp_min(1e-12).log()
            - reference.clamp_min(1e-12).log()
        )
    ).sum(dim=-1)
    if stock_mask is None:
        return per_stock.mean(dim=-1)
    if stock_mask.shape != per_stock.shape:
        raise ValueError("stock_mask must match role stock dimensions")
    weight = stock_mask.to(dtype=per_stock.dtype)
    return (per_stock * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def shrink_role_bias_to_kl_budget(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    budget: float | torch.Tensor,
    stock_mask: torch.Tensor | None = None,
    bisection_steps: int = 16,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Causally shrink a role bias to a per-day categorical-KL budget.

    Every leading batch element is treated as an independent query day.  The
    scalar line search for one day therefore cannot depend on stocks or
    features from any later day in the same batch.  The returned scale is
    detached; gradients still pass through the final scaled bias.  A training
    loss may combine ``raw_kl`` with the returned differentiable ``budget`` to
    train an alarm-aware budget surrogate.
    """

    if logits.shape != bias.shape:
        raise ValueError("role logits and broadcast bias must have matching shapes")
    if logits.ndim < 2:
        raise ValueError("role logits must end in [stock, role]")
    if int(bisection_steps) <= 0:
        raise ValueError("bisection_steps must be positive")
    daily_budget = _normalise_daily_budget(budget, logits=logits)
    reference = torch.softmax(logits, dim=-1)
    raw_roles = torch.softmax(logits + bias, dim=-1)
    raw_kl = _masked_daily_role_kl(raw_roles, reference, stock_mask)

    # The KL along softmax(logits + lambda * bias) is monotone for lambda >= 0.
    # Search every day independently and keep the solver out of autograd.
    with torch.no_grad():
        low = torch.zeros_like(raw_kl)
        high = torch.ones_like(raw_kl)
        feasible_at_one = raw_kl.detach() <= daily_budget.detach()
        low = torch.where(feasible_at_one, torch.ones_like(low), low)
        for _ in range(int(bisection_steps)):
            midpoint = 0.5 * (low + high)
            scale_shape = (*midpoint.shape, 1, 1)
            candidate = torch.softmax(
                logits.detach() + midpoint.reshape(scale_shape) * bias.detach(),
                dim=-1,
            )
            candidate_kl = _masked_daily_role_kl(
                candidate,
                reference.detach(),
                stock_mask,
            )
            feasible = candidate_kl <= daily_budget.detach()
            low = torch.where(feasible, midpoint, low)
            high = torch.where(feasible, high, midpoint)
        scale = torch.where(feasible_at_one, torch.ones_like(low), low)

    scale_shape = (*scale.shape, 1, 1)
    scaled_bias = bias * scale.reshape(scale_shape)
    final_roles = torch.softmax(logits + scaled_bias, dim=-1)
    final_kl = _masked_daily_role_kl(final_roles, reference, stock_mask)
    return scaled_bias, {
        "base_roles": reference,
        "raw_kl": raw_kl,
        "final_kl": final_kl,
        "scale": scale,
        "budget": daily_budget,
    }


def _sender_role_value_adjustment(
    adjustment: torch.Tensor | None,
    *,
    sender_roles: torch.Tensor,
    z: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    """Mix a role-by-feature payload into per-sender stock values.

    Only a single episode-level ``[role, feature]`` tensor is accepted.  This
    prevents a support controller from becoming a hidden per-stock predictor;
    it can influence a stock only through its existing sender-role mixture.
    """

    if adjustment is None:
        return None
    expected = (sender_roles.shape[-1], z.shape[-1])
    if adjustment.ndim != 2 or tuple(adjustment.shape) != expected:
        raise ValueError(f"{name} must have shape [role, feature]={expected}")
    value = adjustment.to(dtype=z.dtype, device=z.device)
    return torch.einsum("...nr,rd->...nd", sender_roles, value)


class DirectedRoleGraph(nn.Module):
    """Role graph with a strict receiver-row/sender-column API.

    For representations ``Z [..., N, d]`` this module constructs daily sender
    and receiver role probabilities and returns

    ``A = softmax_sender(R @ B @ S.T / tau_a)``.

    Consequently ``A[..., i, j]`` carries information from sender stock ``j``
    into receiver stock ``i`` and ``A @ value(Z)`` is the correct aggregation.
    """

    def __init__(
        self,
        d_model: int,
        n_roles: int,
        *,
        tau_s: float = 1.0,
        tau_r: float = 1.0,
        tau_a: float = 1.0,
        role_mode: str = "separate",
        core_mode: str = "directed",
    ) -> None:
        super().__init__()
        if min(tau_s, tau_r, tau_a) <= 0:
            raise ValueError("all temperatures must be positive")
        if role_mode not in {"separate", "tied"}:
            raise ValueError("role_mode must be separate or tied")
        if core_mode not in {"directed", "symmetric"}:
            raise ValueError("core_mode must be directed or symmetric")
        if role_mode == "tied" and abs(float(tau_s) - float(tau_r)) > 1e-12:
            raise ValueError("tied roles require tau_s and tau_r to match")
        self.tau_s = float(tau_s)
        self.tau_r = float(tau_r)
        self.tau_a = float(tau_a)
        self.role_mode = str(role_mode)
        self.core_mode = str(core_mode)
        self.sender_projection = nn.Linear(d_model, n_roles, bias=False)
        # Keep the legacy receiver key even for the tied control.  Aliasing it
        # to the sender module would make state-dict loading order-dependent;
        # freezing an unused receiver module keeps old checkpoint schemas safe.
        self.receiver_projection = nn.Linear(d_model, n_roles, bias=False)
        if self.role_mode == "tied":
            self.receiver_projection.requires_grad_(False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.B0 = nn.Parameter(torch.eye(n_roles))

    def active_core(self, core: torch.Tensor | None = None) -> torch.Tensor:
        """Return the trainable core projected into the requested structure."""

        active = self.B0 if core is None else core
        if self.core_mode == "symmetric":
            active = 0.5 * (active + active.transpose(-1, -2))
        return active

    def forward(
        self,
        z: torch.Tensor,
        *,
        core: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
        sender_role_bias: torch.Tensor | None = None,
        receiver_role_bias: torch.Tensor | None = None,
        sender_role_kl_budget: float | torch.Tensor | None = None,
        receiver_role_kl_budget: float | torch.Tensor | None = None,
        role_kl_bisection_steps: int = 16,
        sender_value_scale: torch.Tensor | None = None,
        sender_value_shift: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if z.ndim < 2:
            raise ValueError("z must end in [stock, feature]")
        if stock_mask is not None and stock_mask.shape != z.shape[:-1]:
            raise ValueError("stock_mask must match z stock dimensions")
        sender_base_logits = self.sender_projection(z) / self.tau_s
        sender_bias = _broadcast_role_bias(sender_role_bias, logits=sender_base_logits)
        sender_base = torch.softmax(sender_base_logits, dim=-1)
        sender_kl = sender_base_logits.new_zeros(sender_base_logits.shape[:-2])
        sender_raw_kl = sender_kl
        sender_scale = sender_kl.new_ones(sender_kl.shape)
        if sender_bias is not None and sender_role_kl_budget is not None:
            sender_bias, sender_trust = shrink_role_bias_to_kl_budget(
                sender_base_logits,
                sender_bias.expand_as(sender_base_logits),
                budget=sender_role_kl_budget,
                stock_mask=stock_mask,
                bisection_steps=role_kl_bisection_steps,
            )
            sender_base = sender_trust["base_roles"]
            sender_raw_kl = sender_trust["raw_kl"]
            sender_kl = sender_trust["final_kl"]
            sender_scale = sender_trust["scale"]
        elif sender_bias is not None:
            sender_unshrunk = torch.softmax(sender_base_logits + sender_bias, dim=-1)
            sender_raw_kl = _masked_daily_role_kl(
                sender_unshrunk, sender_base, stock_mask
            )
            sender_kl = sender_raw_kl
        sender_logits = (
            sender_base_logits if sender_bias is None else sender_base_logits + sender_bias
        )
        sender = torch.softmax(sender_logits, dim=-1)
        if self.role_mode == "tied":
            if receiver_role_bias is not None or receiver_role_kl_budget is not None:
                raise ValueError("tied roles cannot accept an independent receiver route")
            receiver = sender
            receiver_base = sender_base
            receiver_raw_kl = sender_raw_kl
            receiver_kl = sender_kl
            receiver_scale = sender_scale
        else:
            receiver_base_logits = self.receiver_projection(z) / self.tau_r
            receiver_bias = _broadcast_role_bias(
                receiver_role_bias, logits=receiver_base_logits
            )
            receiver_base = torch.softmax(receiver_base_logits, dim=-1)
            receiver_kl = receiver_base_logits.new_zeros(
                receiver_base_logits.shape[:-2]
            )
            receiver_raw_kl = receiver_kl
            receiver_scale = receiver_kl.new_ones(receiver_kl.shape)
            if receiver_bias is not None and receiver_role_kl_budget is not None:
                receiver_bias, receiver_trust = shrink_role_bias_to_kl_budget(
                    receiver_base_logits,
                    receiver_bias.expand_as(receiver_base_logits),
                    budget=receiver_role_kl_budget,
                    stock_mask=stock_mask,
                    bisection_steps=role_kl_bisection_steps,
                )
                receiver_base = receiver_trust["base_roles"]
                receiver_raw_kl = receiver_trust["raw_kl"]
                receiver_kl = receiver_trust["final_kl"]
                receiver_scale = receiver_trust["scale"]
            elif receiver_bias is not None:
                receiver_unshrunk = torch.softmax(
                    receiver_base_logits + receiver_bias, dim=-1
                )
                receiver_raw_kl = _masked_daily_role_kl(
                    receiver_unshrunk, receiver_base, stock_mask
                )
                receiver_kl = receiver_raw_kl
            receiver_logits = (
                receiver_base_logits
                if receiver_bias is None
                else receiver_base_logits + receiver_bias
            )
            receiver = torch.softmax(receiver_logits, dim=-1)
        if stock_mask is not None:
            valid = stock_mask.unsqueeze(-1).to(z.dtype)
            sender = sender * valid
            receiver = receiver * valid
            sender_base = sender_base * valid
            receiver_base = receiver_base * valid

        active_core = self.active_core(core)
        scores = role_scores(receiver, active_core, sender)
        adjacency = sender_softmax(scores, self.tau_a, stock_mask)
        # Role/topology construction is complete before FiLM is applied.  The
        # support-conditioned value payload therefore cannot alter S, R, B,
        # scores, or A; it can only alter what each sender communicates through
        # the already-directed adjacency.
        scale = _sender_role_value_adjustment(
            sender_value_scale,
            sender_roles=sender,
            z=z,
            name="sender_value_scale",
        )
        shift = _sender_role_value_adjustment(
            sender_value_shift,
            sender_roles=sender,
            z=z,
            name="sender_value_shift",
        )
        value_input = z
        if scale is not None:
            value_input = value_input * (1.0 + scale)
        if shift is not None:
            value_input = value_input + shift
        message = adjacency @ self.value_projection(value_input)
        return {
            "A": adjacency,
            "S": sender,
            "R": receiver,
            "S0": sender_base,
            "R0": receiver_base,
            "sender_route_raw_kl": sender_raw_kl,
            "sender_route_kl": sender_kl,
            "sender_route_kl_scale": sender_scale,
            "receiver_route_raw_kl": receiver_raw_kl,
            "receiver_route_kl": receiver_kl,
            "receiver_route_kl_scale": receiver_scale,
            "core": active_core,
            "scores": scores,
            "value_input": value_input,
            "message": message,
        }
