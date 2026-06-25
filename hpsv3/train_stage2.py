"""
HPSv3 Semi-Supervised Adaptive STD Training (ExpSS)

Combines labeled data (pairwise uncertainty/BT loss) and unlabeled data (adaptive std + ratio
constraint), continuing training on top of the OGD checkpoint.

Objectives:
1. The labeled pairwise loss anchors the ranking direction (keeps HPDv3 accuracy >= 0.76).
2. The unlabeled adaptive std loss learns the [model capability, RL iteration] differences.
3. The cross-model ranking loss guarantees that good models score higher than bad models.
4. Ratio constraint: higher capability / higher iteration gets a larger std increase.

Loss formula:
    L_total = L_unsup_total + sup_weight * L_sup
    where L_unsup_total comes from the parent class AdaptiveSTDTrainer:
        rank_weight * L_rank + std_weight * L_std + adaptive_weight * L_adaptive
        + constraint_weight * L_constraint + L_l2
"""

import copy
import json
import os
from datetime import datetime
from functools import partial
from types import SimpleNamespace

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

from hpsv3.dataset.conditioned_rollout_dataset import (
    MixedConditionRankingDataset,
    MixedConditionSTDDataset,
)
from hpsv3.dataset.data_collator_qwen import QWen2VLDataCollator
from hpsv3.dataset.pairwise_dataset import PairwiseOriginalDataset
from hpsv3.model.qwen2vl_trainer import (
    VLMRewardTrainer,
    _convert_A_B_to_chosen_rejected,
    compute_multi_attr_accuracy,
)
from hpsv3.trainer.common import (
    WandbTrainLossCallback,
    create_model_and_processor,
    get_run_timestamp,
    maybe_init_dist,
    set_requires_grad,
)
from hpsv3.trainer.rollout import (
    CsvLoggingCallback,
    RolloutRewardTrainer,
    load_config,
)
from hpsv3.trainer.semisup import InfiniteDataLoaderIterator, create_eval_train_subset
from hpsv3.trainer.adaptive import AdaptiveSTDTrainer, _has_group_encoder, _can_use_group_ids
from hpsv3.utils.adaptive_std import (
    compute_adaptive_std_terms,
    compute_allpair_adaptive_loss,
    compute_conditional_std_bound_loss,
    compute_conditional_std_ratio_bound_loss,
)
from hpsv3.utils.implicit_conditioning import get_model_forward_strip_keys
from hpsv3.utils.parser import ModelConfig, PEFTLoraConfig
from hpsv3.utils.training_utils import load_model_from_checkpoint
from transformers import TrainingArguments

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# Semi-Supervised Adaptive STD Trainer
# ============================================================


class SemiSupAdaptiveSTDTrainer(AdaptiveSTDTrainer):
    """Semi-supervised adaptive STD trainer.

    On top of AdaptiveSTDTrainer (unlabeled adaptive std + ranking + ratio constraint), it adds a
    labeled pairwise DataLoader and computes an extra supervised loss every step to anchor the
    ranking direction.

    Key fix: base_rm (Qwen2VLRewardModelBT)'s forward() does not accept
    iter_values/cond_values/group_ids, so these fields must be stripped before calling the model.
    Both the parent AdaptiveSTDTrainer.compute_loss and _compute_ranking_loss pass these fields,
    so they are fully overridden here.
    """

    # Keys stripped by default (a subclass or __init__ may override)
    _MODEL_EXTRA_KEYS = {"iter_values", "group_ids", "level_ids"}

    def __init__(
        self,
        *args,
        labeled_dataloader: DataLoader = None,
        sup_weight: float = 5.0,
        labeled_loss_type: str = "uncertainty",
        sup_warmup_steps: int = 0,
        **kwargs,
    ):
        adaptive_mode = kwargs.pop("adaptive_mode", "default")
        model_extra_keys_override = kwargs.pop("model_extra_keys", None)
        adaptive_use_predicted_cond = kwargs.pop("adaptive_use_predicted_cond", False)
        predicted_cond_supervision_weight = kwargs.pop("predicted_cond_supervision_weight", 0.0)
        predicted_cond_supervision_loss_type = kwargs.pop("predicted_cond_supervision_loss_type", "smooth_l1")
        super().__init__(*args, **kwargs)
        self._labeled_dataloader = labeled_dataloader
        self._labeled_iter = None
        self.sup_weight = sup_weight
        self.labeled_loss_type = labeled_loss_type
        self.sup_warmup_steps = sup_warmup_steps
        self._sup_count = 0
        self.adaptive_mode = adaptive_mode
        self.adaptive_use_predicted_cond = adaptive_use_predicted_cond
        self.predicted_cond_supervision_weight = predicted_cond_supervision_weight
        self.predicted_cond_supervision_loss_type = predicted_cond_supervision_loss_type
        # film_implicit needs group_ids and cond_values, so only strip iter_values/level_ids
        if model_extra_keys_override is not None:
            self._model_extra_keys = set(model_extra_keys_override)
        else:
            self._model_extra_keys = self._MODEL_EXTRA_KEYS

    def _strip_extra_keys(self, batch_kwargs: dict) -> dict:
        """Strip keys that the model forward does not recognize."""
        return {
            k: v
            for k, v in batch_kwargs.items()
            if k not in self._model_extra_keys
        }

    def _get_labeled_batch(self):
        """Get one batch from the labeled data (infinite cycling)."""
        if self._labeled_iter is None:
            if self._labeled_dataloader is None:
                raise RuntimeError("labeled_dataloader is None")
            self._labeled_iter = InfiniteDataLoaderIterator(self._labeled_dataloader)
        batch = next(self._labeled_iter)
        return self._prepare_inputs(batch)

    def _compute_supervised_loss(self, model, labeled_batch):
        """Compute the loss using labeled pairwise data.

        Temporarily switch loss_type to labeled_loss_type and call
        VLMRewardTrainer.compute_loss (the pairwise branch).
        """
        original_loss_type = self.loss_type
        self.loss_type = self.labeled_loss_type
        try:
            result = VLMRewardTrainer.compute_loss(
                self, model, labeled_batch, return_outputs=False
            )
        finally:
            self.loss_type = original_loss_type
        return result

    def _compute_ranking_loss(self, model, inputs):
        """Override the parent: strip kwargs that base_rm does not recognize before calling the model."""
        batch_1_kwargs = self._strip_extra_keys(dict(inputs["batch_1"]))
        batch_2_kwargs = self._strip_extra_keys(dict(inputs["batch_2"]))

        rewards_A = model(return_dict=True, **batch_1_kwargs)["logits"]
        rewards_B = model(return_dict=True, **batch_2_kwargs)["logits"]

        rewards_chosen, rewards_rejected, nontied_mask = _convert_A_B_to_chosen_rejected(
            rewards_A,
            rewards_B,
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
        """Full override: combines the unlabeled adaptive std and labeled pairwise loss.

        Reason for overriding: the parent AdaptiveSTDTrainer.compute_loss passes
        iter_values/cond_values/group_ids to the model forward, but base_rm does not accept these
        arguments. Here we strip them before calling the model; the rest of the loss computation
        matches the parent.
        """
        step = self.state.global_step if hasattr(self, "state") else 0

        # ---- 1. Ranking loss (cross-model good > bad) ----
        rank_scale = 1.0 if self._use_rank_loss_this_step() else 0.0
        rank_loss = None
        if rank_scale > 0.0:
            rank_batch = self._get_ranking_batch()
            rank_loss = self._compute_ranking_loss(model, rank_batch)

        # ---- 2. STD forward (strip kwargs the model does not recognize) ----
        batch_kwargs = self._strip_extra_keys(dict(inputs["batch_all"]))
        # -- inject conditional fields from inputs top-level --
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

        # ---- 3. Per-group std + adaptive + constraint ----
        group_stds = self._extract_group_stds(rewards, inputs["k_per_prompt"])
        group_cond_values = self._extract_group_cond_values(inputs, rewards.device)
        group_ogd_stds = self._extract_group_ogd_std(inputs, rewards.device)

        # Implicit model: use the model-inferred cond instead of the hardcoded cond
        pred_cond_sup_loss = torch.zeros((), device=rewards.device)
        if self.adaptive_use_predicted_cond and isinstance(outputs, dict):
            cond_pred = outputs.get("cond_pred")
            if cond_pred is not None and cond_pred.dim() == 2:
                # Aggregate to group level (take the first value of each group, since inference is identical within a group)
                pred_group_cond = self._extract_group_cond_values(
                    {"cond_values": cond_pred, "k_per_prompt": inputs["k_per_prompt"]},
                    rewards.device,
                )
                # Supervision loss: inferred cond ~= ground-truth cond
                if self.predicted_cond_supervision_weight > 0:
                    from hpsv3.utils.adaptive_std import compute_predicted_condition_supervision_loss
                    pred_cond_sup_loss = compute_predicted_condition_supervision_loss(
                        pred_group_cond, group_cond_values,
                        loss_type=self.predicted_cond_supervision_loss_type,
                    )
                # Use the inferred cond for adaptive/constraint (instead of the hardcoded one)
                group_cond_values = pred_group_cond

        std_loss, adaptive_loss = compute_adaptive_std_terms(
            group_stds=group_stds,
            group_ogd_stds=group_ogd_stds,
            group_cond_values=group_cond_values,
            adaptive_margin=self.adaptive_margin,
            std_floor=self.std_floor,
            adaptive_priority_mode=self.adaptive_priority_mode,
            eps=self.adaptive_eps,
        )

        # all-pairs adaptive mode: denser gradient signal
        if self.adaptive_mode == "allpairs" and group_stds.numel() >= 2:
            adaptive_loss = compute_allpair_adaptive_loss(
                group_stds=group_stds,
                group_ogd_stds=group_ogd_stds,
                group_cond_values=group_cond_values,
                margin=self.adaptive_margin,
                eps=self.adaptive_eps,
            )

        # Constraint loss
        constraint_loss = torch.zeros((), device=rewards.device)
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
                constraint_loss, ratio_vals, lower, upper, target = (
                    compute_conditional_std_ratio_bound_loss(
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
                )
                ratio_mean = ratio_vals.mean()
        else:
            if self.std_constraint_enable and constraint_weight > 0.0:
                constraint_loss, lower, upper, target = (
                    compute_conditional_std_bound_loss(
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
                )
                ratio_mean = ratio_vals.mean()

        # L2 regularization
        l2_coef = getattr(self.args, "reward_l2_coef", 0.03)
        l2_loss = l2_coef * rewards.pow(2).mean()

        # Scale
        scale = 0.0 if step < self.warmup_steps else 1.0
        constraint_scale = 0.0 if step < constraint_warmup_steps else 1.0

        unsup_total = (
            rank_scale * self.rank_weight * rank_loss
            + scale * self.std_weight * std_loss
            + scale * self.adaptive_weight * adaptive_loss
            + constraint_scale * constraint_weight * constraint_loss
            + scale * self.predicted_cond_supervision_weight * pred_cond_sup_loss
            + l2_loss
        )

        # ---- 4. Labeled supervised pairwise loss ----
        sup_scale = 0.0 if step < self.sup_warmup_steps else 1.0
        if sup_scale > 0.0 and self._labeled_dataloader is not None:
            labeled_batch = self._get_labeled_batch()
            sup_loss = self._compute_supervised_loss(model, labeled_batch)
            self._sup_count += 1
        else:
            sup_loss = torch.zeros((), device=rewards.device)

        # ---- 5. Total loss ----
        total = unsup_total + sup_scale * self.sup_weight * sup_loss

        # ---- 6. Logging ----
        if step % 10 == 0:
            print(
                f"[SemiSupAdaptive] step={step} "
                f"rank={rank_loss.item():.4f}(x{rank_scale:.0f}) "
                f"std={std_loss.item():.4f} adpt={adaptive_loss.item():.4f} "
                f"cstr={constraint_loss.item():.4f}(x{constraint_scale:.0f}) "
                f"sup={sup_loss.item():.4f}(x{sup_scale * self.sup_weight:.1f}) "
                f"ratio={ratio_mean.item():.4f} total={total.item():.4f}"
            )

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "ss/rank_loss": rank_loss.detach().item(),
                    "ss/std_loss": std_loss.detach().item(),
                    "ss/adaptive_loss": adaptive_loss.detach().item(),
                    "ss/constraint_loss": constraint_loss.detach().item(),
                    "ss/sup_loss": sup_loss.detach().item(),
                    "ss/l2_loss": l2_loss.detach().item(),
                    "ss/unsup_total": unsup_total.detach().item(),
                    "ss/total_loss": total.detach().item(),
                    "ss/group_std_mean": group_stds.detach().mean().item(),
                    "ss/group_ratio_mean": ratio_mean.detach().item(),
                    "ss/std_lower_mean": lower.detach().mean().item(),
                    "ss/std_upper_mean": upper.detach().mean().item(),
                    "ss/sup_count": self._sup_count,
                },
                step=step,
            )

        if return_outputs:
            return total, {}
        return total

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """At eval time, go straight through the pairwise logic (inherited from CombinedStage2Trainer)."""
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )


# ============================================================
# Main Training Function
# ============================================================


def main(config, local_rank=0, debug=False):
    """Main entry point for semi-supervised adaptive STD training."""
    args = load_config(config)

    # Initialize distributed training
    maybe_init_dist()
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    is_main = rank == 0

    timestamp = get_run_timestamp()
    exp_name = os.environ.get("EXP_NAME") or "semisup_adaptive_std"
    output_dir = os.path.join(
        args.output_dir, f"HPSv3_7B_{exp_name}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Key parameters
    metadata_path = getattr(args, "metadata_path", None)
    rl_iter_metadata_path = getattr(args, "rl_iter_metadata_path", None)
    ogd_std_path = getattr(args, "ogd_std_path", "")
    tiers = getattr(args, "tiers", ["sd15", "sdxl", "qwen_image"])
    labeled_json_list = getattr(args, "labeled_json_list", [])
    # Open-source CSV entry: when the config provides rollout_json/labeled_json/data_root,
    # train directly from the released long-table CSV (paths relative to data_root), with the same
    # behavior as the original JSON.
    rollout_json = getattr(args, "rollout_json", None)
    labeled_json = getattr(args, "labeled_json", None)
    data_root = getattr(args, "data_root", None)
    sup_weight = getattr(args, "sup_weight", 5.0)
    labeled_loss_type = getattr(args, "labeled_loss_type", "uncertainty")
    sup_warmup_steps = getattr(args, "sup_warmup_steps", 0)

    if is_main:
        print(f"{'=' * 60}")
        print("HPSv3 Semi-Supervised Adaptive STD Training")
        print(f"{'=' * 60}")
        print(f"Metadata: {metadata_path}")
        print(f"RL iter: {rl_iter_metadata_path}")
        print(f"OGD std: {ogd_std_path}")
        print(f"Labeled: {labeled_json_list}")
        print(f"Sup weight: {sup_weight}, Loss: {labeled_loss_type}")
        print(f"Output: {output_dir}")
        print(f"{'=' * 60}\n")

    # --------------------------------------------------------
    # Model creation
    # --------------------------------------------------------
    model_type = getattr(args, "model_type", "base_rm")
    cond_dim = getattr(args, "cond_dim", 256)

    model_config = ModelConfig(
        model_name_or_path=args.model_name_or_path,
        rm_head_type=getattr(args, "rm_head_type", "ranknet"),
        output_dim=getattr(args, "output_dim", 2),
        use_special_tokens=getattr(args, "use_special_tokens", True),
        reward_token=getattr(args, "reward_token", "special"),
        loss_type=getattr(args, "loss_type", "max_std_unsup"),
        freeze_vision_tower=getattr(args, "freeze_vision_tower", True),
        freeze_llm=getattr(args, "freeze_llm", True),
        tune_merger=getattr(args, "tune_merger", False),
        torch_dtype=getattr(args, "torch_dtype", "bfloat16"),
    )

    peft_lora_config = PEFTLoraConfig(
        lora_enable=getattr(args, "lora_enable", False),
        vision_lora=getattr(args, "vision_lora", False),
        lora_r=getattr(args, "lora_r", 512),
        lora_alpha=getattr(args, "lora_alpha", 1024),
        num_lora_modules=getattr(args, "num_lora_modules", -1),
    )

    _training_args_ns = SimpleNamespace(
        bf16=getattr(args, "bf16", True),
        fp16=False,
        disable_flash_attn2=getattr(args, "disable_flash_attn2", False),
    )

    if model_type == "film_hybrid":
        from hpsv3.trainer.adaptive import create_film_hybrid_model_and_processor

        if is_main:
            print(f"[Model] FiLMHybrid (cond_dim={cond_dim}, CapabilityEncoder + explicit iter)")
        model, processor, peft_config = create_film_hybrid_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=_training_args_ns,
            cond_dim=cond_dim,
        )
    else:
        if is_main:
            print("[Model] Base RM (Qwen2VLRewardModelBT)")
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=_training_args_ns,
        )

    # Load the OGD checkpoint
    if hasattr(args, "load_from_pretrained") and args.load_from_pretrained:
        if is_main:
            print(f"Loading checkpoint: {args.load_from_pretrained}")
        model, loaded_step = load_model_from_checkpoint(
            model,
            args.load_from_pretrained,
            getattr(args, "load_from_pretrained_step", None),
        )
        if is_main:
            print(f"Loaded step: {loaded_step}")

    # --------------------------------------------------------
    # Freeze logic
    # --------------------------------------------------------
    freeze_llm = getattr(args, "freeze_llm", True)
    freeze_vit = getattr(args, "freeze_vision_tower", True)
    tune_merger = getattr(args, "tune_merger", False)
    train_rm_head = getattr(args, "train_rm_head", True)

    # Detect Qwen3VL vs Qwen2VL model architecture
    _is_qwen3vl = hasattr(model.model, "language_model")
    if _is_qwen3vl:
        _llm_module = model.model.language_model
        _vit_module = model.model.visual
    else:
        _llm_module = model.model
        _vit_module = model.visual

    if freeze_llm or freeze_vit:
        if freeze_llm:
            for p in _llm_module.parameters():
                p.requires_grad_(False)
        if freeze_vit:
            for p in _vit_module.parameters():
                p.requires_grad_(False)
        if tune_merger and freeze_vit:
            for p in _vit_module.merger.parameters():
                p.requires_grad_(True)

        # rm_head is always trainable
        if train_rm_head and hasattr(model, "rm_head"):
            for p in model.rm_head.parameters():
                p.requires_grad_(True)

        # FiLMContinuous: cond_encoder and film_gen must be trainable
        for cond_name in ("cond_encoder", "film_gen", "level_embedding", "iter_proj", "group_encoder", "cap_encoder", "cap_cond_encoder", "cap_film_gen", "iter_cond_encoder", "iter_film_gen", "cap_residual_head"):
            if hasattr(model, cond_name):
                for p in getattr(model, cond_name).parameters():
                    p.requires_grad_(True)

        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        if is_main:
            print(
                f"[Freeze] frozen={total_params - trainable}/{total_params}, "
                f"trainable={trainable}"
            )


    # --------------------------------------------------------
    # Load the pretrained CapabilityEncoder (film_hybrid)
    # --------------------------------------------------------
    cap_encoder_path = getattr(args, "pretrained_cap_encoder", None)
    if cap_encoder_path and hasattr(model, "cap_encoder"):
        state_dict = torch.load(cap_encoder_path, map_location="cpu")
        model.cap_encoder.load_state_dict(state_dict)
        if is_main:
            print(f"[CapEncoder] Loaded pretrained: {cap_encoder_path}")
    # Set per-image cap inference mode (v5+)
    if getattr(args, "per_image_cap", False) and hasattr(model, "per_image_cap"):
        model.per_image_cap = True
    elif getattr(args, "per_image_cap", False):
        model.per_image_cap = True
    freeze_cap_encoder = getattr(args, "freeze_cap_encoder", False)
    if freeze_cap_encoder and hasattr(model, "cap_encoder"):
        for p in model.cap_encoder.parameters():
            p.requires_grad_(False)
        if is_main:
            print("[CapEncoder] Frozen")
    # --------------------------------------------------------
    # Data Collator
    # --------------------------------------------------------
    data_collator = QWen2VLDataCollator(
        processor=processor,
        max_pixels=getattr(args, "max_pixels", 200704),
        min_pixels=getattr(args, "min_pixels", 200704),
    )

    # --------------------------------------------------------
    # Unlabeled data (rollout, the main dataset)
    # --------------------------------------------------------
    if is_main:
        print("Creating mixed-condition STD dataset (unlabeled)...")
    train_dataset = MixedConditionSTDDataset(
        metadata_path=metadata_path,
        rl_iter_metadata_path=rl_iter_metadata_path,
        ogd_std_path=ogd_std_path,
        json_path=rollout_json,
        data_root=data_root,
        tiers=tiers,
        max_images_per_group=getattr(args, "max_images_per_group", 6),
        val_size=getattr(args, "val_size", 500),
        split="train",
        rl_tier=getattr(args, "rl_tier", None),
        max_rl_step=getattr(args, "max_rl_step", None),
    )
    if is_main:
        print(f"Unlabeled STD dataset: {len(train_dataset)} groups")

    # --------------------------------------------------------
    # Cross-model ranking data (unlabeled, good model > bad model)
    # --------------------------------------------------------
    if is_main:
        print("Creating mixed-condition ranking dataset...")
    ranking_dataset = MixedConditionRankingDataset(
        metadata_path=metadata_path,
        rl_iter_metadata_path=rl_iter_metadata_path,
        json_path=rollout_json,
        data_root=data_root,
        tiers=tiers,
        pairs_per_prompt=getattr(args, "pairs_per_prompt", 3),
        pair_mode=getattr(args, "ranking_pair_mode", "cross_model_only"),
        rl_tier=getattr(args, "rl_tier", None),
        max_rl_step=getattr(args, "max_rl_step", None),
    )
    if is_main:
        print(f"Ranking dataset: {len(ranking_dataset)} pairs")

    ranking_sampler = (
        DistributedSampler(ranking_dataset, shuffle=True) if is_distributed else None
    )
    ranking_dataloader = DataLoader(
        ranking_dataset,
        batch_size=getattr(args, "per_device_train_batch_size", 1),
        sampler=ranking_sampler,
        shuffle=(ranking_sampler is None),
        collate_fn=data_collator,
        num_workers=getattr(args, "dataloader_num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Labeled data (pairwise)
    # --------------------------------------------------------
    labeled_dataloader = None
    if labeled_json or labeled_json_list:
        if is_main:
            print("Creating labeled pairwise dataset...")
        if labeled_json:
            labeled_dataset = PairwiseOriginalDataset(
                data_json_list=[labeled_json], data_root=data_root)
        else:
            labeled_dataset = PairwiseOriginalDataset(json_list=labeled_json_list)
        if is_main:
            print(f"Labeled dataset: {len(labeled_dataset)} pairs")

        labeled_sampler = (
            DistributedSampler(labeled_dataset, shuffle=True)
            if is_distributed
            else None
        )
        labeled_dataloader = DataLoader(
            labeled_dataset,
            batch_size=getattr(args, "per_device_train_batch_size", 1),
            sampler=labeled_sampler,
            shuffle=(labeled_sampler is None),
            collate_fn=data_collator,
            num_workers=getattr(args, "dataloader_num_workers", 2),
            pin_memory=True,
            drop_last=True,
        )

    # --------------------------------------------------------
    # Evaluation datasets
    # --------------------------------------------------------
    eval_datasets = {}
    # Open-source CSV eval entry: config provides test_json_list=[[name, csv_path], ...]
    test_json_list = getattr(args, "test_json_list", None)
    if test_json_list:
        for test_name, test_json in test_json_list:
            eval_datasets[test_name] = PairwiseOriginalDataset(
                data_json_list=[test_json], data_root=data_root)
            if is_main:
                print(f"Eval [{test_name}]: {len(eval_datasets[test_name])} pairs")
    elif hasattr(args, "test_json_list") and args.test_json_list:
        for item in args.test_json_list:
            test_name, test_paths = item
            eval_datasets[test_name] = PairwiseOriginalDataset(json_list=test_paths)
            if is_main:
                print(f"Eval [{test_name}]: {len(eval_datasets[test_name])} pairs")

    # Training-data subset evaluation
    eval_train_json = getattr(args, "eval_train_json", None)
    eval_train_size = getattr(args, "eval_train_size", 10000)
    if eval_train_json:
        eval_train_subset = create_eval_train_subset(
            labeled_json_list=[eval_train_json],
            eval_train_size=eval_train_size,
            output_dir=output_dir,
            is_main=is_main,
        )
        eval_datasets["Own Aesthetic Train (10k)"] = eval_train_subset
        if is_main:
            print(
                f"Eval [Own Aesthetic Train (10k)]: {len(eval_train_subset)} pairs"
            )

    # --------------------------------------------------------
    # Training Arguments
    # --------------------------------------------------------
    wandb_run_name = os.environ.get("WANDB_NAME", exp_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=wandb_run_name,
        num_train_epochs=getattr(args, "num_train_epochs", 3),
        max_steps=getattr(args, "max_steps", -1),
        per_device_train_batch_size=getattr(args, "per_device_train_batch_size", 1),
        per_device_eval_batch_size=getattr(args, "per_device_eval_batch_size", 8),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 4),
        learning_rate=getattr(args, "learning_rate", 4e-5),
        warmup_ratio=getattr(args, "warmup_ratio", 0.05),
        lr_scheduler_type=getattr(args, "lr_scheduler_type", "cosine"),
        logging_steps=getattr(args, "logging_steps", 10),
        eval_strategy=getattr(args, "eval_strategy", "steps"),
        eval_steps=getattr(args, "eval_steps", 500),
        save_steps=getattr(args, "save_steps", 500),
        save_total_limit=None,
        bf16=getattr(args, "bf16", True),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=getattr(args, "dataloader_num_workers", 2),
        remove_unused_columns=False,
        report_to=getattr(args, "report_to", "wandb"),
        deepspeed=(
            getattr(args, "deepspeed", None)
            if hasattr(args, "deepspeed")
            else None
        ),
    )

    # Custom parameters
    training_args.vision_lr = getattr(args, "vision_lr", None)
    training_args.merger_lr = getattr(args, "merger_lr", None)
    training_args.rm_head_lr = getattr(args, "rm_head_lr", None)
    training_args.special_token_lr = getattr(args, "special_token_lr", None)
    training_args.local_rank = local_rank
    training_args.save_full_model = getattr(args, "save_full_model", True)
    training_args.save_only_model = getattr(args, "save_only_model", True)
    training_args.vis_steps = getattr(args, "vis_steps", 200)
    training_args.reward_l2_coef = getattr(args, "reward_l2_coef", 0.03)
    training_args.kl_coef = getattr(args, "kl_coef", 0.0)

    training_args.disable_dropout = getattr(args, "disable_dropout", False)
    training_args.dataset_num_proc = getattr(args, "dataset_num_proc", None)
    training_args.max_length = getattr(args, "max_length", 512)
    # save_epochs -> save_steps
    actual_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )
    if hasattr(args, "save_epochs") and args.save_epochs:
        training_args.save_steps = round(
            args.save_epochs * len(train_dataset) / actual_batch_size
        )

    # --------------------------------------------------------
    # Reference Model (KL)
    # --------------------------------------------------------
    ref_model = None
    kl_coef = getattr(args, "kl_coef", 0.0)
    if kl_coef > 0.0:
        if is_main:
            print("Loading frozen reference model for KL constraint...")
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if is_main:
            print("Reference model ready.")

    # --------------------------------------------------------
    # Create the Trainer
    # --------------------------------------------------------
    use_conditioned_model = model_type in ("film_continuous", "film_implicit", "film_hybrid", "dual_film", "iter_film_cap_residual")

    trainer = SemiSupAdaptiveSTDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if eval_datasets else None,
        data_collator=data_collator,
        compute_metrics=partial(compute_multi_attr_accuracy),
        # From RolloutRewardTrainer
        loss_type=getattr(args, "loss_type", "max_std_unsup"),
        ref_model=ref_model,
        # From CombinedStage2Trainer
        ranking_dataloader=ranking_dataloader,
        ranking_loss_type=getattr(args, "ranking_loss_type", "bt"),
        std_weight=getattr(args, "std_weight", 1.0),
        rank_weight=getattr(args, "rank_weight", 2.0),
        ranking_margin=getattr(args, "ranking_margin", 0.0),
        rank_step_interval=getattr(args, "rank_step_interval", 1),
        rank_warmup_steps=getattr(args, "rank_warmup_steps", 0),
        # From ConditionedSTDTrainer
        use_conditioned_model=use_conditioned_model,
        # From AdaptiveSTDTrainer
        ogd_std_path=ogd_std_path,
        adaptive_weight=getattr(args, "adaptive_weight", 1.5),
        adaptive_margin=getattr(args, "adaptive_margin", 0.35),
        adaptive_priority_mode=getattr(
            args, "adaptive_priority_mode", "strong_cap_high_iter"
        ),
        warmup_steps=getattr(args, "warmup_steps", 500),
        std_floor=getattr(args, "std_floor", -2.0),
        adaptive_eps=getattr(args, "adaptive_eps", 1e-6),
        std_constraint_enable=getattr(args, "std_constraint_enable", True),
        std_constraint_weight=getattr(args, "std_constraint_weight", 1.2),
        std_constraint_warmup_steps=getattr(args, "std_constraint_warmup_steps", 500),
        std_constraint_mode=getattr(args, "std_constraint_mode", "absolute"),
        std_bound_lower_base=getattr(args, "std_bound_lower_base", 0.08),
        std_bound_lower_cap_coef=getattr(args, "std_bound_lower_cap_coef", 0.08),
        std_bound_lower_iter_coef=getattr(args, "std_bound_lower_iter_coef", 0.55),
        std_bound_lower_ogd_coef=getattr(args, "std_bound_lower_ogd_coef", 0.50),
        std_bound_upper_base=getattr(args, "std_bound_upper_base", 0.35),
        std_bound_upper_cap_coef=getattr(args, "std_bound_upper_cap_coef", 0.15),
        std_bound_upper_iter_coef=getattr(args, "std_bound_upper_iter_coef", 0.55),
        std_bound_upper_ogd_coef=getattr(args, "std_bound_upper_ogd_coef", 1.25),
        std_bound_min=getattr(args, "std_bound_min", 0.05),
        std_bound_min_gap=getattr(args, "std_bound_min_gap", 0.06),
        std_target_enable=getattr(args, "std_target_enable", True),
        std_target_weight=getattr(args, "std_target_weight", 0.8),
        std_target_base=getattr(args, "std_target_base", 0.15),
        std_target_cap_coef=getattr(args, "std_target_cap_coef", 0.10),
        std_target_iter_coef=getattr(args, "std_target_iter_coef", 0.60),
        std_target_ogd_coef=getattr(args, "std_target_ogd_coef", 0.95),
        std_target_margin_base=getattr(args, "std_target_margin_base", 0.30),
        std_target_margin_cap_coef=getattr(args, "std_target_margin_cap_coef", 0.05),
        std_target_margin_iter_coef=getattr(args, "std_target_margin_iter_coef", 0.20),
        # std_ratio_* constraint params
        std_ratio_constraint_enable=getattr(args, "std_ratio_constraint_enable", False),
        std_ratio_constraint_weight=getattr(args, "std_ratio_constraint_weight", 0.0),
        std_ratio_constraint_warmup_steps=getattr(args, "std_ratio_constraint_warmup_steps", 0),
        std_ratio_lower_base=getattr(args, "std_ratio_lower_base", 1.0),
        std_ratio_lower_cap_coef=getattr(args, "std_ratio_lower_cap_coef", 0.0),
        std_ratio_lower_iter_coef=getattr(args, "std_ratio_lower_iter_coef", 0.0),
        std_ratio_upper_base=getattr(args, "std_ratio_upper_base", 1.5),
        std_ratio_upper_cap_coef=getattr(args, "std_ratio_upper_cap_coef", 0.0),
        std_ratio_upper_iter_coef=getattr(args, "std_ratio_upper_iter_coef", 0.0),
        std_ratio_min=getattr(args, "std_ratio_min", 0.0),
        std_ratio_min_gap=getattr(args, "std_ratio_min_gap", 1e-4),
        std_ratio_target_enable=getattr(args, "std_ratio_target_enable", False),
        std_ratio_target_weight=getattr(args, "std_ratio_target_weight", 1.0),
        std_ratio_target_base=getattr(args, "std_ratio_target_base", 1.0),
        std_ratio_target_cap_coef=getattr(args, "std_ratio_target_cap_coef", 0.0),
        std_ratio_target_iter_coef=getattr(args, "std_ratio_target_iter_coef", 0.0),
        std_ratio_target_margin_base=getattr(args, "std_ratio_target_margin_base", 0.0),
        std_ratio_target_margin_cap_coef=getattr(args, "std_ratio_target_margin_cap_coef", 0.0),
        std_ratio_target_margin_iter_coef=getattr(args, "std_ratio_target_margin_iter_coef", 0.0),
        std_ratio_space=getattr(args, "std_ratio_space", "raw"),
        # New: labeled supervised loss
        labeled_dataloader=labeled_dataloader,
        sup_weight=sup_weight,
        labeled_loss_type=labeled_loss_type,
        sup_warmup_steps=sup_warmup_steps,
        adaptive_mode=getattr(args, "adaptive_mode", "default"),
        adaptive_use_predicted_cond=getattr(args, "adaptive_use_predicted_cond", False),
        predicted_cond_supervision_weight=getattr(args, "predicted_cond_supervision_weight", 0.0),
        predicted_cond_supervision_loss_type=getattr(args, "predicted_cond_supervision_loss_type", "smooth_l1"),
        # implicit setup must not see GT cond_values in forward;
        # GT cond stays in inputs for supervision/adaptive losses only.
        model_extra_keys=get_model_forward_strip_keys(model_type),
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------
    if wandb and getattr(args, "report_to", "none") == "wandb":
        trainer.add_callback(WandbTrainLossCallback())

    csv_log_path = os.path.join(output_dir, "training_log.csv")
    trainer.add_callback(CsvLoggingCallback(log_path=csv_log_path))

    if is_main:
        print(f"[CSV log] {csv_log_path}")

    # --------------------------------------------------------
    # Save the config
    # --------------------------------------------------------
    if is_main:
        config_save_path = os.path.join(output_dir, "config.json")

        def namespace_to_dict(obj):
            if isinstance(obj, SimpleNamespace):
                return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, list):
                return [namespace_to_dict(item) for item in obj]
            else:
                return obj

        with open(config_save_path, "w") as f:
            json.dump(namespace_to_dict(args), f, indent=2)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    if is_main:
        print(f"\n{'=' * 60}")
        print("Starting semi-supervised adaptive STD training...")
        print(f"{'=' * 60}\n")

    trainer.train()

    # Save the final model
    if is_main:
        print(f"\n{'=' * 60}")
        print("Training completed!")
        print(f"Saving final model to {output_dir}")
        print(f"{'=' * 60}\n")

    trainer.save_model(output_dir)

    if is_main:
        print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
