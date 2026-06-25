"""Utilities for ExpY adaptive std losses."""

import torch
import torch.nn.functional as F


def compute_adaptive_std_terms(
    group_stds: torch.Tensor,
    group_ogd_stds: torch.Tensor,
    group_cond_values: torch.Tensor,
    adaptive_margin: float = 0.1,
    std_floor: float = -2.0,
    adaptive_priority_mode: str = "strong_cap_high_iter",
    eps: float = 1e-6,
):
    """Compute L_std (clamped) and L_adaptive for ExpY."""
    std_loss = torch.clamp(-torch.log1p(group_stds), min=std_floor).mean()

    if group_stds.numel() < 2:
        return std_loss, torch.zeros((), device=group_stds.device)

    ratios = group_stds / torch.clamp(group_ogd_stds, min=eps)
    cap = group_cond_values[:, 0]
    iter_val = group_cond_values[:, 1]

    mode = str(adaptive_priority_mode or "strong_cap_high_iter").lower()
    if mode == "strong_cap_high_iter":
        priority = cap + iter_val
    elif mode == "low_cap_high_iter":
        priority = (1.0 - cap) + iter_val
    elif mode == "cap_only":
        priority = cap
    elif mode == "iter_only":
        priority = iter_val
    else:
        raise ValueError(f"Unsupported adaptive_priority_mode: {adaptive_priority_mode}")

    high_idx = torch.argmax(priority)
    low_idx = torch.argmin(priority)
    adaptive_loss = F.relu(ratios[low_idx] - ratios[high_idx] + adaptive_margin)
    return std_loss, adaptive_loss


def compute_allpair_adaptive_loss(
    group_stds: torch.Tensor,
    group_ogd_stds: torch.Tensor,
    group_cond_values: torch.Tensor,
    margin: float = 0.35,
    eps: float = 1e-6,
):
    """All-pairs adaptive loss: for every pair where priority[i] > priority[j],
    enforce ratio[i] > ratio[j] - margin.

    Provides a denser gradient signal than only taking the two argmax/argmin extremes.
    With batch_size=4 there are C(4,2)=6 pairs.
    """
    if group_stds.numel() < 2:
        return torch.zeros((), device=group_stds.device)

    ratios = group_stds / torch.clamp(group_ogd_stds, min=eps)
    priority = group_cond_values[:, 0] + group_cond_values[:, 1]

    n = len(ratios)
    # Vectorized: build the priority diff and ratio diff for all pairs
    p_diff = priority.unsqueeze(0) - priority.unsqueeze(1)  # [n, n], >0 means row has higher priority
    r_diff = ratios.unsqueeze(1) - ratios.unsqueeze(0)      # [n, n], >0 means col has larger ratio
    # For pairs with p_diff > 0: require ratio[row] > ratio[col] - margin
    # i.e. loss = relu(ratio[col] - ratio[row] + margin) = relu(r_diff + margin)
    mask = (p_diff > 0).float()
    pair_losses = F.relu(r_diff + margin) * mask
    count = mask.sum()
    if count > 0:
        return pair_losses.sum() / count
    return torch.zeros((), device=group_stds.device)


def _linear_condition_term(
    group_cond_values: torch.Tensor,
    base: float,
    cap_coef: float,
    iter_coef: float,
) -> torch.Tensor:
    """Compute a linear term from condition [capability, iter]."""
    cap = group_cond_values[:, 0]
    iter_val = group_cond_values[:, 1]
    return base + cap_coef * cap + iter_coef * iter_val


def compute_conditional_std_bound_loss(
    group_stds: torch.Tensor,
    group_cond_values: torch.Tensor,
    group_ogd_stds: torch.Tensor,
    lower_base: float = 0.0,
    lower_cap_coef: float = 0.0,
    lower_iter_coef: float = 0.0,
    lower_ogd_coef: float = 0.0,
    upper_base: float = 10.0,
    upper_cap_coef: float = 0.0,
    upper_iter_coef: float = 0.0,
    upper_ogd_coef: float = 0.0,
    bound_min: float = 0.0,
    min_gap: float = 1e-4,
    use_target: bool = False,
    target_weight: float = 1.0,
    target_base: float = 0.0,
    target_cap_coef: float = 0.0,
    target_iter_coef: float = 0.0,
    target_ogd_coef: float = 0.0,
    target_margin_base: float = 0.0,
    target_margin_cap_coef: float = 0.0,
    target_margin_iter_coef: float = 0.0,
):
    """Compute explicit condition-aware std constraint loss.

    Returns:
        total_loss: scalar tensor.
        lower: [G] lower bounds.
        upper: [G] upper bounds.
        target: [G] target std (for logging; center of bounds if target disabled).
    """
    lower = _linear_condition_term(
        group_cond_values, lower_base, lower_cap_coef, lower_iter_coef
    ) + lower_ogd_coef * group_ogd_stds
    lower = torch.clamp(lower, min=bound_min)

    upper = _linear_condition_term(
        group_cond_values, upper_base, upper_cap_coef, upper_iter_coef
    ) + upper_ogd_coef * group_ogd_stds
    upper = torch.maximum(upper, lower + min_gap)

    under_penalty = F.relu(lower - group_stds)
    over_penalty = F.relu(group_stds - upper)
    bound_loss = (under_penalty + over_penalty).mean()

    target = 0.5 * (lower + upper)
    if not use_target:
        return bound_loss, lower, upper, target

    target = _linear_condition_term(
        group_cond_values, target_base, target_cap_coef, target_iter_coef
    ) + target_ogd_coef * group_ogd_stds
    target_margin = _linear_condition_term(
        group_cond_values, target_margin_base, target_margin_cap_coef, target_margin_iter_coef
    )
    target_margin = torch.clamp(target_margin, min=0.0)
    target_penalty = F.relu(torch.abs(group_stds - target) - target_margin).mean()
    total = bound_loss + target_weight * target_penalty
    return total, lower, upper, target


def compute_conditional_std_ratio_bound_loss(
    group_stds: torch.Tensor,
    group_cond_values: torch.Tensor,
    group_ogd_stds: torch.Tensor,
    lower_base: float = 1.0,
    lower_cap_coef: float = 0.0,
    lower_iter_coef: float = 0.0,
    upper_base: float = 1.5,
    upper_cap_coef: float = 0.0,
    upper_iter_coef: float = 0.0,
    ratio_min: float = 0.0,
    min_gap: float = 1e-4,
    use_target: bool = False,
    target_weight: float = 1.0,
    target_base: float = 1.0,
    target_cap_coef: float = 0.0,
    target_iter_coef: float = 0.0,
    target_margin_base: float = 0.0,
    target_margin_cap_coef: float = 0.0,
    target_margin_iter_coef: float = 0.0,
    ratio_space: str = "raw",
    eps: float = 1e-6,
):
    """Compute condition-aware ratio constraints on std growth over OGD.

    ratio = std_new / std_ogd

    Returns:
        total_loss: scalar tensor.
        ratio: [G] ratio values.
        lower_ratio: [G] lower ratio bounds.
        upper_ratio: [G] upper ratio bounds.
        target_ratio: [G] target ratio (for logging; center of bounds if target disabled).
    """
    ratio_raw = group_stds / torch.clamp(group_ogd_stds, min=eps)

    lower_ratio_raw = _linear_condition_term(
        group_cond_values, lower_base, lower_cap_coef, lower_iter_coef
    )
    lower_ratio_raw = torch.clamp(lower_ratio_raw, min=ratio_min)

    upper_ratio_raw = _linear_condition_term(
        group_cond_values, upper_base, upper_cap_coef, upper_iter_coef
    )
    upper_ratio_raw = torch.maximum(upper_ratio_raw, lower_ratio_raw + min_gap)

    ratio_mode = str(ratio_space or "raw").lower()
    if ratio_mode == "raw":
        ratio_cmp = ratio_raw
        lower_cmp = lower_ratio_raw
        upper_cmp = upper_ratio_raw
    elif ratio_mode == "log":
        ratio_cmp = torch.log(torch.clamp(ratio_raw, min=eps))
        lower_cmp = torch.log(torch.clamp(lower_ratio_raw, min=eps))
        upper_cmp = torch.log(torch.clamp(upper_ratio_raw, min=eps))
    else:
        raise ValueError(f"Unsupported ratio_space: {ratio_space}")

    under_penalty = F.relu(lower_cmp - ratio_cmp)
    over_penalty = F.relu(ratio_cmp - upper_cmp)
    bound_loss = (under_penalty + over_penalty).mean()

    target_ratio_raw = 0.5 * (lower_ratio_raw + upper_ratio_raw)
    if not use_target:
        return bound_loss, ratio_raw, lower_ratio_raw, upper_ratio_raw, target_ratio_raw

    target_ratio_raw = _linear_condition_term(
        group_cond_values, target_base, target_cap_coef, target_iter_coef
    )
    target_margin = _linear_condition_term(
        group_cond_values, target_margin_base, target_margin_cap_coef, target_margin_iter_coef
    )
    target_margin = torch.clamp(target_margin, min=0.0)
    if ratio_mode == "raw":
        target_penalty = F.relu(torch.abs(ratio_raw - target_ratio_raw) - target_margin).mean()
    else:
        target_low = torch.log(torch.clamp(target_ratio_raw - target_margin, min=eps))
        target_high = torch.log(torch.clamp(target_ratio_raw + target_margin, min=eps))
        target_penalty = (
            F.relu(target_low - ratio_cmp) + F.relu(ratio_cmp - target_high)
        ).mean()
    total = bound_loss + target_weight * target_penalty
    return total, ratio_raw, lower_ratio_raw, upper_ratio_raw, target_ratio_raw


def compute_predicted_condition_supervision_loss(
    pred_cond_values: torch.Tensor,
    target_cond_values: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Supervise predicted implicit condition with explicit condition labels."""
    if pred_cond_values.numel() == 0:
        return torch.zeros((), device=pred_cond_values.device)

    pred = pred_cond_values.float()
    target = target_cond_values.to(device=pred.device, dtype=torch.float32)
    mode = str(loss_type or "smooth_l1").lower()
    if mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    if mode == "mse":
        return F.mse_loss(pred, target)
    if mode == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(f"Unsupported predicted condition supervision loss: {loss_type}")
