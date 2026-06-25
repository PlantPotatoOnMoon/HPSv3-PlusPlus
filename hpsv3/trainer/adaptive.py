"""
HPSv3++ Stage 2 conditioned-training building blocks.

Provides the adaptive-STD trainer hierarchy
(ConditionedSTDTrainer -> CombinedStage2Trainer -> AdaptiveSTDTrainer) and the
FiLM-hybrid model factory (create_film_hybrid_model_and_processor). These are
consumed by hpsv3/train_stage2.py (training) and hpsv3/inference.py (the
film_hybrid reward-model path); this module is not a standalone entry point.
"""

import json
import os
import inspect

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hpsv3.model.qwen2vl_trainer import (
    Qwen2VLRewardModelBT,
    _convert_A_B_to_chosen_rejected,
)
from hpsv3.utils.adaptive_std import (
    compute_adaptive_std_terms,
    compute_conditional_std_bound_loss,
    compute_conditional_std_ratio_bound_loss,
    compute_predicted_condition_supervision_loss,
)
from hpsv3.trainer.rollout import (
    RolloutRewardTrainer,
)

from transformers import AutoProcessor
from trl import get_kbit_device_map, get_quantization_config

try:
    import wandb
except ImportError:
    wandb = None

try:
    import flash_attn
except ImportError:
    flash_attn = None


# ============================================================
# Conditioned Trainers
# ============================================================


def _has_group_encoder(model) -> bool:
    """True if model (or wrapped model.module) has implicit group encoder."""
    unwrapped = getattr(model, "module", model)
    return hasattr(unwrapped, "group_encoder") or hasattr(unwrapped, "cap_encoder")


def _supports_group_ids(model) -> bool:
    """True if model forward explicitly accepts group_ids."""
    unwrapped = getattr(model, "module", model)
    forward_fn = getattr(unwrapped, "forward", None)
    if forward_fn is None:
        return False
    try:
        sig = inspect.signature(forward_fn)
    except (TypeError, ValueError):
        return False
    return "group_ids" in sig.parameters


def _can_use_group_ids(model) -> bool:
    return _has_group_encoder(model) and _supports_group_ids(model)



class ConditionedSTDTrainer(RolloutRewardTrainer):
    """STD-mode trainer, supports embedding condition.

    Exp2 (cond_emb + std): uses Qwen2VLRewardModelConditioned,
         passing level_ids at forward time
    Exp4 (cond_text + std): uses the plain Qwen2VLRewardModelBT,
         condition is already in the text
    """

    def __init__(self, *args, use_conditioned_model=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_conditioned_model = use_conditioned_model

    def _compute_std_loss(self, model, inputs, return_outputs=False):
        """Override to support level_ids in forward."""
        batch_kwargs = dict(inputs["batch_all"])
        if self.use_conditioned_model and "level_ids" in inputs:
            batch_kwargs["level_ids"] = inputs["level_ids"]
        if "iter_values" in inputs:
            batch_kwargs["iter_values"] = inputs["iter_values"]
        if "cond_values" in inputs and not _has_group_encoder(model):
            batch_kwargs["cond_values"] = inputs["cond_values"]
        if "group_ids" in inputs and _can_use_group_ids(model):
            batch_kwargs["group_ids"] = inputs["group_ids"]
        rewards = model(return_dict=True, **batch_kwargs)["logits"]

        r = rewards[:, 0]

        stds = []
        groups = []
        start = 0
        for k in inputs["k_per_prompt"]:
            groups.append((start, k))
            stds.append(r[start: start + k].std(unbiased=False))
            start += k
        std_batch = torch.stack(stds)

        # L2 reg
        l2_coef = getattr(self.args, "reward_l2_coef", 0.01)
        l2_reg = l2_coef * r.pow(2).mean()

        if self.loss_type == "max_std_unsup":
            loss = -std_batch.mean() + l2_reg
        else:
            loss = std_batch.mean() + l2_reg

        # KL constraint
        kl_coef = getattr(self.args, "kl_coef", 0.0)
        kl_loss = torch.tensor(0.0, device=r.device)
        if kl_coef > 0.0 and self.ref_model is not None:
            if not self._ref_on_device:
                self.ref_model = self.ref_model.to(r.device)
                self._ref_on_device = True
            with torch.no_grad():
                if self.use_conditioned_model and "level_ids" in inputs:
                    ref_kwargs = dict(inputs["batch_all"])
                    ref_kwargs["level_ids"] = inputs["level_ids"]
                    if "iter_values" in inputs:
                        ref_kwargs["iter_values"] = inputs["iter_values"]
                    if "cond_values" in inputs and not _has_group_encoder(self.ref_model):
                        ref_kwargs["cond_values"] = inputs["cond_values"]
                    if "group_ids" in inputs and _can_use_group_ids(self.ref_model):
                        ref_kwargs["group_ids"] = inputs["group_ids"]
                    r_ref = self.ref_model(return_dict=True, **ref_kwargs)["logits"][:, 0]
                else:
                    ref_kwargs = dict(inputs["batch_all"])
                    if "iter_values" in inputs:
                        ref_kwargs["iter_values"] = inputs["iter_values"]
                    if "cond_values" in inputs and not _has_group_encoder(self.ref_model):
                        ref_kwargs["cond_values"] = inputs["cond_values"]
                    if "group_ids" in inputs and _can_use_group_ids(self.ref_model):
                        ref_kwargs["group_ids"] = inputs["group_ids"]
                    r_ref = self.ref_model(return_dict=True, **ref_kwargs)["logits"][:, 0]
            kl_loss = kl_coef * (r - r_ref).pow(2).mean()
            loss = loss + kl_loss

        if wandb is not None and wandb.run is not None:
            step = self.state.global_step if hasattr(self, "state") else 0
            r_det = r.detach()
            log = {
                "rollout/reward_std": std_batch.detach().mean().item(),
                "rollout/reward_mean": r_det.mean().item(),
                "rollout/reward_max": r_det.max().item(),
                "rollout/reward_min": r_det.min().item(),
                "rollout/l2_reg": l2_reg.detach().item(),
            }
            if kl_coef > 0.0:
                log["rollout/kl_loss"] = kl_loss.detach().item()
            wandb.log(log, step=step)

        if return_outputs:
            return loss, {"rewards_all": rewards}
        return loss


class CombinedStage2Trainer(ConditionedSTDTrainer):
    """Combined-mode trainer: jointly optimizes per-tier STD + cross-tier Ranking.

    Inherits ConditionedSTDTrainer (which already provides per-tier std capability),
    and additionally holds a separate ranking DataLoader, computing both losses at each step.

    Modeled on the mixed_loss mode of SemiSupRewardTrainer.
    """

    def __init__(
        self,
        *args,
        ranking_dataloader: DataLoader = None,
        ranking_loss_type: str = "bt",
        std_weight: float = 1.0,
        rank_weight: float = 0.5,
        ranking_margin: float = 2.0,
        rank_step_interval: int = 1,
        rank_warmup_steps: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._ranking_dataloader = ranking_dataloader
        self._ranking_iter = None
        self.ranking_loss_type = ranking_loss_type
        self.std_weight = std_weight
        self.rank_weight = rank_weight
        self.ranking_margin = ranking_margin
        self.rank_step_interval = max(int(rank_step_interval), 1)
        self.rank_warmup_steps = max(int(rank_warmup_steps), 0)

    def _get_ranking_batch(self):
        """Fetch one batch from the ranking DataLoader (infinite loop)."""
        if self._ranking_iter is None:
            if self._ranking_dataloader is None:
                raise RuntimeError("ranking_dataloader is None")
            from hpsv3.trainer.semisup import InfiniteDataLoaderIterator
            self._ranking_iter = InfiniteDataLoaderIterator(self._ranking_dataloader)
        batch = next(self._ranking_iter)
        return self._prepare_inputs(batch)

    def _use_rank_loss_this_step(self) -> bool:
        step = self.state.global_step if hasattr(self, "state") else 0
        if step < self.rank_warmup_steps:
            return False
        return ((step - self.rank_warmup_steps) % self.rank_step_interval) == 0

    def _compute_ranking_loss(self, model, inputs):
        """Compute the ranking (BT) loss, supports embedding condition."""
        batch_1_kwargs = dict(inputs["batch_1"])
        batch_2_kwargs = dict(inputs["batch_2"])
        if self.use_conditioned_model and "level_ids_1" in inputs:
            batch_1_kwargs["level_ids"] = inputs["level_ids_1"]
            batch_2_kwargs["level_ids"] = inputs["level_ids_2"]
        if "iter_values_1" in inputs:
            batch_1_kwargs["iter_values"] = inputs["iter_values_1"]
            batch_2_kwargs["iter_values"] = inputs["iter_values_2"]
        if "cond_values_1" in inputs and not _has_group_encoder(model):
            batch_1_kwargs["cond_values"] = inputs["cond_values_1"]
            batch_2_kwargs["cond_values"] = inputs["cond_values_2"]
        if _can_use_group_ids(model):
            bs = batch_1_kwargs["input_ids"].shape[0]
            device = batch_1_kwargs["input_ids"].device
            group_ids = torch.arange(bs, dtype=torch.long, device=device)
            batch_1_kwargs["group_ids"] = group_ids
            batch_2_kwargs["group_ids"] = group_ids

        rewards_A = model(return_dict=True, **batch_1_kwargs)["logits"]
        rewards_B = model(return_dict=True, **batch_2_kwargs)["logits"]

        from hpsv3.model.qwen2vl_trainer import _convert_A_B_to_chosen_rejected
        (rewards_chosen, rewards_rejected, nontied_mask) = _convert_A_B_to_chosen_rejected(
            rewards_A, rewards_B,
            tied_threshold=getattr(self, "tied_threshold", 0.0),
            choice_dist=inputs["choice_dist"],
        )

        if self.ranking_loss_type == "margin_bt":
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - self.ranking_margin
            )
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)
        loss = (loss * nontied_mask).mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute per-tier STD loss + ranking loss together at each step."""
        # 1. Per-tier STD loss (from the train_dataset batch)
        std_loss = self._compute_std_loss(model, inputs, return_outputs=False)

        # 2. Ranking loss (batch taken from the separate DataLoader)
        rank_scale = 1.0 if self._use_rank_loss_this_step() else 0.0
        if rank_scale > 0.0:
            rank_batch = self._get_ranking_batch()
            rank_loss = self._compute_ranking_loss(model, rank_batch)
        else:
            rank_loss = torch.zeros((), device=std_loss.device)

        # 3. Weighted combination
        total_loss = self.std_weight * std_loss + rank_scale * self.rank_weight * rank_loss

        if wandb is not None and wandb.run is not None:
            step = self.state.global_step if hasattr(self, "state") else 0
            wandb.log({
                "combined/std_loss": std_loss.detach().item(),
                "combined/rank_loss": rank_loss.detach().item(),
                "combined/rank_scale": float(rank_scale),
                "combined/total_loss": total_loss.detach().item(),
            }, step=step)

        if self.state.global_step % 50 == 0:
            print(f"[Combined] std={std_loss.item():.4f} "
                  f"rank={rank_loss.item():.4f} total={total_loss.item():.4f}")

        if return_outputs:
            return total_loss, {}
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """At eval time, go straight through the pairwise logic using the correct level."""
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(
                getattr(self.model, "config", None), "keys_to_ignore_at_inference", []
            ) or []

        # Inject level_ids into the batch so eval uses the correct level
        if self.use_conditioned_model and "level_ids_1" in inputs:
            if "batch_1" in inputs:
                inputs["batch_1"]["level_ids"] = inputs["level_ids_1"]
            if "batch_2" in inputs:
                inputs["batch_2"]["level_ids"] = inputs["level_ids_2"]

        with torch.no_grad():
            loss, logits_dict = super(RolloutRewardTrainer, self).compute_loss(
                model, inputs, return_outputs=True
            )

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        if "rewards_A" in logits_dict and "rewards_B" in logits_dict:
            r_A = logits_dict["rewards_A"][:, [0]]
            r_B = logits_dict["rewards_B"][:, [0]]
            logits = torch.cat([r_A, r_B], dim=1).detach()
        else:
            logits = torch.zeros((1, 2), device=loss.device)

        labels = torch.ones((logits.shape[0], 1), device=logits.device)
        return loss, logits, labels


class AdaptiveSTDTrainer(CombinedStage2Trainer):
    """ExpY: L_rank + L_std(clamp) + L_adaptive(OGD improvement ratio)."""

    def __init__(
        self,
        *args,
        ogd_std_path: str = "",
        adaptive_weight: float = 2.0,
        adaptive_margin: float = 0.1,
        adaptive_priority_mode: str = "strong_cap_high_iter",
        warmup_steps: int = 300,
        std_floor: float = -2.0,
        adaptive_eps: float = 1e-6,
        adaptive_use_predicted_cond: bool = False,
        std_constraint_enable: bool = False,
        std_constraint_weight: float = 0.0,
        std_constraint_warmup_steps: int = 0,
        std_constraint_mode: str = "absolute",
        std_bound_lower_base: float = 0.0,
        std_bound_lower_cap_coef: float = 0.0,
        std_bound_lower_iter_coef: float = 0.0,
        std_bound_lower_ogd_coef: float = 0.0,
        std_bound_upper_base: float = 10.0,
        std_bound_upper_cap_coef: float = 0.0,
        std_bound_upper_iter_coef: float = 0.0,
        std_bound_upper_ogd_coef: float = 0.0,
        std_bound_min: float = 0.0,
        std_bound_min_gap: float = 1e-4,
        std_target_enable: bool = False,
        std_target_weight: float = 1.0,
        std_target_base: float = 0.0,
        std_target_cap_coef: float = 0.0,
        std_target_iter_coef: float = 0.0,
        std_target_ogd_coef: float = 0.0,
        std_target_margin_base: float = 0.0,
        std_target_margin_cap_coef: float = 0.0,
        std_target_margin_iter_coef: float = 0.0,
        std_ratio_constraint_enable: bool = False,
        std_ratio_constraint_weight: float = 0.0,
        std_ratio_constraint_warmup_steps: int = 0,
        std_ratio_lower_base: float = 1.0,
        std_ratio_lower_cap_coef: float = 0.0,
        std_ratio_lower_iter_coef: float = 0.0,
        std_ratio_upper_base: float = 1.5,
        std_ratio_upper_cap_coef: float = 0.0,
        std_ratio_upper_iter_coef: float = 0.0,
        std_ratio_min: float = 0.0,
        std_ratio_min_gap: float = 1e-4,
        std_ratio_target_enable: bool = False,
        std_ratio_target_weight: float = 1.0,
        std_ratio_target_base: float = 1.0,
        std_ratio_target_cap_coef: float = 0.0,
        std_ratio_target_iter_coef: float = 0.0,
        std_ratio_target_margin_base: float = 0.0,
        std_ratio_target_margin_cap_coef: float = 0.0,
        std_ratio_target_margin_iter_coef: float = 0.0,
        std_ratio_space: str = "raw",
        predicted_cond_supervision_weight: float = 0.0,
        predicted_cond_supervision_warmup_steps: int = 0,
        predicted_cond_supervision_loss: str = "smooth_l1",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adaptive_weight = adaptive_weight
        self.adaptive_margin = adaptive_margin
        self.adaptive_priority_mode = str(adaptive_priority_mode or "strong_cap_high_iter")
        self.warmup_steps = warmup_steps
        self.std_floor = std_floor
        self.adaptive_eps = adaptive_eps
        self.adaptive_use_predicted_cond = adaptive_use_predicted_cond
        self.ogd_std_map = {}
        if ogd_std_path and os.path.exists(ogd_std_path):
            with open(ogd_std_path, "r") as f:
                self.ogd_std_map = json.load(f)
        self.ogd_std_path = ogd_std_path
        self.std_constraint_enable = std_constraint_enable
        self.std_constraint_weight = std_constraint_weight
        self.std_constraint_warmup_steps = std_constraint_warmup_steps
        self.std_constraint_mode = str(std_constraint_mode or "absolute").lower()
        self.std_bound_lower_base = std_bound_lower_base
        self.std_bound_lower_cap_coef = std_bound_lower_cap_coef
        self.std_bound_lower_iter_coef = std_bound_lower_iter_coef
        self.std_bound_lower_ogd_coef = std_bound_lower_ogd_coef
        self.std_bound_upper_base = std_bound_upper_base
        self.std_bound_upper_cap_coef = std_bound_upper_cap_coef
        self.std_bound_upper_iter_coef = std_bound_upper_iter_coef
        self.std_bound_upper_ogd_coef = std_bound_upper_ogd_coef
        self.std_bound_min = std_bound_min
        self.std_bound_min_gap = std_bound_min_gap
        self.std_target_enable = std_target_enable
        self.std_target_weight = std_target_weight
        self.std_target_base = std_target_base
        self.std_target_cap_coef = std_target_cap_coef
        self.std_target_iter_coef = std_target_iter_coef
        self.std_target_ogd_coef = std_target_ogd_coef
        self.std_target_margin_base = std_target_margin_base
        self.std_target_margin_cap_coef = std_target_margin_cap_coef
        self.std_target_margin_iter_coef = std_target_margin_iter_coef
        self.std_ratio_constraint_enable = std_ratio_constraint_enable
        self.std_ratio_constraint_weight = std_ratio_constraint_weight
        self.std_ratio_constraint_warmup_steps = std_ratio_constraint_warmup_steps
        self.std_ratio_lower_base = std_ratio_lower_base
        self.std_ratio_lower_cap_coef = std_ratio_lower_cap_coef
        self.std_ratio_lower_iter_coef = std_ratio_lower_iter_coef
        self.std_ratio_upper_base = std_ratio_upper_base
        self.std_ratio_upper_cap_coef = std_ratio_upper_cap_coef
        self.std_ratio_upper_iter_coef = std_ratio_upper_iter_coef
        self.std_ratio_min = std_ratio_min
        self.std_ratio_min_gap = std_ratio_min_gap
        self.std_ratio_target_enable = std_ratio_target_enable
        self.std_ratio_target_weight = std_ratio_target_weight
        self.std_ratio_target_base = std_ratio_target_base
        self.std_ratio_target_cap_coef = std_ratio_target_cap_coef
        self.std_ratio_target_iter_coef = std_ratio_target_iter_coef
        self.std_ratio_target_margin_base = std_ratio_target_margin_base
        self.std_ratio_target_margin_cap_coef = std_ratio_target_margin_cap_coef
        self.std_ratio_target_margin_iter_coef = std_ratio_target_margin_iter_coef
        self.std_ratio_space = str(std_ratio_space or "raw").lower()
        self.predicted_cond_supervision_weight = predicted_cond_supervision_weight
        self.predicted_cond_supervision_warmup_steps = predicted_cond_supervision_warmup_steps
        self.predicted_cond_supervision_loss = str(
            predicted_cond_supervision_loss or "smooth_l1"
        ).lower()

    def _extract_group_stds(self, scores: torch.Tensor, k_per_prompt: list) -> torch.Tensor:
        stds = []
        start = 0
        for k in k_per_prompt:
            stds.append(scores[start: start + k].std(unbiased=False))
            start += k
        return torch.stack(stds)

    @staticmethod
    def _aggregate_group_values(values: torch.Tensor, k_per_prompt: list) -> torch.Tensor:
        groups = []
        start = 0
        for k in k_per_prompt:
            groups.append(values[start: start + k].mean(dim=0))
            start += k
        return torch.stack(groups)

    def _extract_group_cond_values(self, inputs, device) -> torch.Tensor:
        if "cond_values" in inputs:
            cond_flat = inputs["cond_values"].to(device).float()
            conds = []
            start = 0
            for k in inputs["k_per_prompt"]:
                conds.append(cond_flat[start])
                start += k
            return torch.stack(conds)

        if "level_ids" in inputs:
            levels = inputs["level_ids"].to(device)
            if "iter_values" in inputs:
                iter_vals = inputs["iter_values"].to(device).float()
            else:
                iter_vals = torch.zeros_like(levels, dtype=torch.float32)
            level_to_cap = torch.tensor([0.0, 0.5, 1.0], device=device)
            caps = level_to_cap[torch.clamp(levels, min=0, max=2)]
            cond_flat = torch.stack([caps, iter_vals], dim=-1)
            conds = []
            start = 0
            for k in inputs["k_per_prompt"]:
                conds.append(cond_flat[start])
                start += k
            return torch.stack(conds)

        # fallback
        n_group = len(inputs["k_per_prompt"])
        return torch.zeros((n_group, 2), device=device, dtype=torch.float32)

    def _extract_group_ogd_std(self, inputs, device) -> torch.Tensor:
        if "ogd_std" in inputs:
            return inputs["ogd_std"].to(device).float()
        if "group_keys" in inputs and self.ogd_std_map:
            vals = [float(self.ogd_std_map.get(k, 0.0)) for k in inputs["group_keys"]]
            return torch.tensor(vals, device=device, dtype=torch.float32)
        return torch.ones((len(inputs["k_per_prompt"]),), device=device, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rank_scale = 1.0 if self._use_rank_loss_this_step() else 0.0
        rank_loss = None
        if rank_scale > 0.0:
            rank_batch = self._get_ranking_batch()
            rank_loss = self._compute_ranking_loss(model, rank_batch)

        batch_kwargs = dict(inputs["batch_all"])
        if self.use_conditioned_model and "level_ids" in inputs:
            batch_kwargs["level_ids"] = inputs["level_ids"]
        if "iter_values" in inputs:
            batch_kwargs["iter_values"] = inputs["iter_values"]
        if "cond_values" in inputs and not _has_group_encoder(model):
            batch_kwargs["cond_values"] = inputs["cond_values"]
        if "group_ids" in inputs and _can_use_group_ids(model):
            batch_kwargs["group_ids"] = inputs["group_ids"]

        outputs = model(return_dict=True, **batch_kwargs)
        rewards = outputs["logits"][:, 0]
        if rank_loss is None:
            rank_loss = torch.zeros((), device=rewards.device)
        group_stds = self._extract_group_stds(rewards, inputs["k_per_prompt"])
        cond_source = "label"
        group_cond_values = None
        if self.adaptive_use_predicted_cond and isinstance(outputs, dict):
            cond_pred = outputs.get("cond_pred")
            if isinstance(cond_pred, torch.Tensor) and cond_pred.dim() == 2 and cond_pred.shape[0] == rewards.shape[0]:
                group_cond_values = self._aggregate_group_values(
                    cond_pred.float(), inputs["k_per_prompt"]
                ).to(rewards.device)
                cond_source = "pred"
        if group_cond_values is None:
            group_cond_values = self._extract_group_cond_values(inputs, rewards.device)
        group_ogd_stds = self._extract_group_ogd_std(inputs, rewards.device)

        std_loss, adaptive_loss = compute_adaptive_std_terms(
            group_stds=group_stds,
            group_ogd_stds=group_ogd_stds,
            group_cond_values=group_cond_values,
            adaptive_margin=self.adaptive_margin,
            std_floor=self.std_floor,
            adaptive_priority_mode=self.adaptive_priority_mode,
            eps=self.adaptive_eps,
        )

        constraint_loss = torch.zeros((), device=rewards.device)
        pred_cond_sup_loss = torch.zeros((), device=rewards.device)
        ratio_vals = group_stds / torch.clamp(group_ogd_stds, min=self.adaptive_eps)
        ratio_mean = ratio_vals.mean()
        lower = torch.zeros_like(group_stds)
        upper = torch.zeros_like(group_stds)
        target = torch.zeros_like(group_stds)
        constraint_weight = self.std_constraint_weight
        constraint_warmup_steps = self.std_constraint_warmup_steps
        if self.std_constraint_mode == "ratio":
            constraint_weight = self.std_ratio_constraint_weight
            constraint_warmup_steps = self.std_ratio_constraint_warmup_steps
            if self.std_ratio_constraint_enable and constraint_weight > 0.0:
                constraint_loss, ratio_vals, lower, upper, target = compute_conditional_std_ratio_bound_loss(
                    group_stds=group_stds,
                    group_cond_values=group_cond_values,
                    group_ogd_stds=group_ogd_stds,
                    lower_base=self.std_ratio_lower_base,
                    lower_cap_coef=self.std_ratio_lower_cap_coef,
                    lower_iter_coef=self.std_ratio_lower_iter_coef,
                    upper_base=self.std_ratio_upper_base,
                    upper_cap_coef=self.std_ratio_upper_cap_coef,
                    upper_iter_coef=self.std_ratio_upper_iter_coef,
                    ratio_min=self.std_ratio_min,
                    min_gap=self.std_ratio_min_gap,
                    use_target=self.std_ratio_target_enable,
                    target_weight=self.std_ratio_target_weight,
                    target_base=self.std_ratio_target_base,
                    target_cap_coef=self.std_ratio_target_cap_coef,
                    target_iter_coef=self.std_ratio_target_iter_coef,
                    target_margin_base=self.std_ratio_target_margin_base,
                    target_margin_cap_coef=self.std_ratio_target_margin_cap_coef,
                    target_margin_iter_coef=self.std_ratio_target_margin_iter_coef,
                    ratio_space=self.std_ratio_space,
                    eps=self.adaptive_eps,
                )
                ratio_mean = ratio_vals.mean()
        else:
            if self.std_constraint_enable and constraint_weight > 0.0:
                constraint_loss, lower, upper, target = compute_conditional_std_bound_loss(
                    group_stds=group_stds,
                    group_cond_values=group_cond_values,
                    group_ogd_stds=group_ogd_stds,
                    lower_base=self.std_bound_lower_base,
                    lower_cap_coef=self.std_bound_lower_cap_coef,
                    lower_iter_coef=self.std_bound_lower_iter_coef,
                    lower_ogd_coef=self.std_bound_lower_ogd_coef,
                    upper_base=self.std_bound_upper_base,
                    upper_cap_coef=self.std_bound_upper_cap_coef,
                    upper_iter_coef=self.std_bound_upper_iter_coef,
                    upper_ogd_coef=self.std_bound_upper_ogd_coef,
                    bound_min=self.std_bound_min,
                    min_gap=self.std_bound_min_gap,
                    use_target=self.std_target_enable,
                    target_weight=self.std_target_weight,
                    target_base=self.std_target_base,
                    target_cap_coef=self.std_target_cap_coef,
                    target_iter_coef=self.std_target_iter_coef,
                    target_ogd_coef=self.std_target_ogd_coef,
                    target_margin_base=self.std_target_margin_base,
                    target_margin_cap_coef=self.std_target_margin_cap_coef,
                    target_margin_iter_coef=self.std_target_margin_iter_coef,
                )
                ratio_mean = ratio_vals.mean()

        l2_coef = getattr(self.args, "reward_l2_coef", 0.05)
        l2_loss = l2_coef * rewards.pow(2).mean()

        scale = 0.0 if self.state.global_step < self.warmup_steps else 1.0
        constraint_scale = (
            0.0 if self.state.global_step < constraint_warmup_steps else 1.0
        )
        pred_cond_sup_scale = (
            0.0
            if self.state.global_step < self.predicted_cond_supervision_warmup_steps
            else 1.0
        )
        if (
            self.predicted_cond_supervision_weight > 0.0
            and isinstance(outputs, dict)
            and "cond_pred" in outputs
            and isinstance(outputs["cond_pred"], torch.Tensor)
            and "cond_values" in inputs
        ):
            pred_cond_sup_loss = compute_predicted_condition_supervision_loss(
                outputs["cond_pred"],
                inputs["cond_values"],
                loss_type=self.predicted_cond_supervision_loss,
            )
        total = (
            rank_scale * self.rank_weight * rank_loss
            + scale * self.std_weight * std_loss
            + scale * self.adaptive_weight * adaptive_loss
            + constraint_scale * constraint_weight * constraint_loss
            + pred_cond_sup_scale
            * self.predicted_cond_supervision_weight
            * pred_cond_sup_loss
            + l2_loss
        )

        if self.state.global_step % 10 == 0:
            print(
                f"[AdaptiveSTD] step={self.state.global_step} "
                f"rank={rank_loss.item():.4f}(x{rank_scale:.1f}) "
                f"std={std_loss.item():.4f}(x{scale:.1f}) "
                f"adaptive={adaptive_loss.item():.4f}(x{scale:.1f}) "
                f"constraint={constraint_loss.item():.4f}(x{constraint_scale:.1f}) "
                f"pred_sup={pred_cond_sup_loss.item():.4f}(x{pred_cond_sup_scale:.1f}) "
                f"ratio_mean={ratio_mean.item():.4f} "
                f"cond_src={cond_source} "
                f"l2={l2_loss.item():.4f} "
                f"total={total.item():.4f}"
            )

        if wandb is not None and wandb.run is not None:
            step = self.state.global_step if hasattr(self, "state") else 0
            wandb.log(
                {
                    "expy/rank_loss": rank_loss.detach().item(),
                    "expy/rank_scale": float(rank_scale),
                    "expy/std_loss": std_loss.detach().item(),
                    "expy/adaptive_loss": adaptive_loss.detach().item(),
                    "expy/constraint_loss": constraint_loss.detach().item(),
                    "expy/pred_cond_sup_loss": pred_cond_sup_loss.detach().item(),
                    "expy/l2_loss": l2_loss.detach().item(),
                    "expy/scale": float(scale),
                    "expy/constraint_scale": float(constraint_scale),
                    "expy/pred_cond_sup_scale": float(pred_cond_sup_scale),
                    "expy/constraint_mode_ratio": 1.0 if self.std_constraint_mode == "ratio" else 0.0,
                    "expy/ratio_space_log": 1.0 if self.std_ratio_space == "log" else 0.0,
                    "expy/total_loss": total.detach().item(),
                    "expy/group_std_mean": group_stds.detach().mean().item(),
                    "expy/group_ratio_mean": ratio_mean.detach().item(),
                    "expy/std_lower_mean": lower.detach().mean().item(),
                    "expy/std_upper_mean": upper.detach().mean().item(),
                    "expy/std_target_mean": target.detach().mean().item(),
                },
                step=step,
            )

        if return_outputs:
            return total, {}
        return total




# ============================================================
# V3 Trainer: L_rank + L_std + L_cal + L_order + L_mono
# ============================================================



# ============================================================
# V4 Adaptive Margin Trainer
# ============================================================



















# ============================================================
# Conditioned Model creation functions
# ============================================================



# ============================================================
# Load pretrained weights into conditioned model
# ============================================================



# ============================================================
# Multi-Head Model creation & loading
# ============================================================





# ============================================================
# Rescaled / FiLM Model creation & loading
# ============================================================





def _create_model_common(cls_name, model_config, peft_lora_config, training_args,
                         extra_kwargs=None):
    """Common model-creation logic, to reduce duplicated code."""
    # Auto-detect Qwen3-VL vs Qwen2-VL based on model_name_or_path
    _mnop = getattr(model_config, "model_name_or_path", "")
    if "qwen3" in _mnop.lower() or "Qwen3" in _mnop:
        from hpsv3.model import qwen3vl_rm as cond_module
        # Map Qwen2VL class names to Qwen3VL equivalents
        _q3_map = {
            "Qwen2VLRewardModelFiLMContinuous": "Qwen3VLRewardModelFiLMContinuous",
            "Qwen2VLRewardModelConditioned": "Qwen3VLRewardModelBT",
            "Qwen2VLRewardModelFiLMHybrid": "Qwen3VLRewardModelFiLMHybrid",
        }
        cls_name = _q3_map.get(cls_name, cls_name)
    else:
        raise ValueError(
            f"Only Qwen3-VL backbones are supported in this release; got {_mnop}"
        )
    ModelCls = getattr(cond_module, cls_name)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False,
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right",
    )
    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    model = ModelCls.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=(
            "flash_attention_2"
            if not training_args.disable_flash_attn2 and flash_attn is not None
            else "sdpa"
        ),
        rm_head_type=model_config.rm_head_type,
        rm_head_kwargs=model_config.rm_head_kwargs,
        **(extra_kwargs or {}),
        **model_kwargs,
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))
    if training_args.bf16:
        model.to(torch.bfloat16)
    model.rm_head.to(torch.float32)
    # Cast all condition-related modules to float32
    for attr in ["level_embedding", "iter_proj", "film_gen", "cond_encoder",
                 "scale_gen", "shift_gen", "cond_head",
                 "attn_proj", "margin_head",
                 "sim_proj", "var_proj", "cross_attn", "key_proj",
                 "pair_margin_head", "group_encoder"]:
        if hasattr(model, attr):
            getattr(model, attr).to(torch.float32)
    # nn.Parameter needs special handling (.to() is not in-place)
    if hasattr(model, "set_query") and isinstance(model.set_query, torch.nn.Parameter):
        model.set_query.data = model.set_query.data.float()
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, None



















def create_film_hybrid_model_and_processor(
    model_config, peft_lora_config, training_args,
    cond_dim=256,
):
    """Create the hybrid-condition FiLM model (implicit CapabilityEncoder + explicit iter)."""
    return _create_model_common(
        "Qwen2VLRewardModelFiLMHybrid", model_config, peft_lora_config, training_args,
        extra_kwargs=dict(cond_dim=cond_dim),
    )
