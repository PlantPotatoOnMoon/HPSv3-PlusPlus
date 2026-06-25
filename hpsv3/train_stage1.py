"""
Gradient Orthogonal Projection (OGD) Continue Learning for HPSv3.

Core idea:
- Each training step computes gradients on new data (our own) and reference data (HPDv3).
- If cos(grad_new, grad_ref) > 0 (acute angle): gradients are compatible, use grad_new directly.
- If cos(grad_new, grad_ref) <= 0 (obtuse angle): project grad_new onto the orthogonal
  complement of grad_ref, removing the conflicting component.

This prevents catastrophic forgetting by ensuring new-task updates never oppose
the reference-task gradient direction.
"""

import json
import os
import math
from datetime import datetime
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Optional, List

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from hpsv3.model.qwen2vl_trainer import (
    Qwen2VLRewardModelBT,
    VLMRewardTrainer,
    compute_multi_attr_accuracy,
    PartialEmbeddingUpdateCallback,
    _convert_A_B_to_chosen_rejected,
)
from hpsv3.dataset.pairwise_dataset import PairwiseOriginalDataset
from hpsv3.dataset.data_collator_qwen import QWen2VLDataCollator
from hpsv3.utils.parser import ModelConfig, PEFTLoraConfig, TrainingConfig, DataConfig
from hpsv3.utils.training_utils import load_model_from_checkpoint, find_target_linear_names
from hpsv3.utils.parser import parse_args_with_yaml
from transformers import AutoProcessor, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import get_kbit_device_map, get_quantization_config
from hpsv3.model.differentiable_image_processor import Qwen2VLImageProcessor

try:
    import wandb
except ImportError:
    wandb = None
try:
    import flash_attn
except ImportError:
    flash_attn = None
    print("Flash Attention is not installed. Falling to SDPA.")

import socket


# --- Import shared utilities from train.py ---
from hpsv3.trainer.common import (
    maybe_init_dist,
    get_run_timestamp,
    create_model_and_processor,
    save_configs_to_json,
    set_requires_grad,
    WandbTrainLossCallback,
)


# --- OGD Trainer: override training_step to do gradient projection ---

class OGDRewardTrainer(VLMRewardTrainer):
    """
    Extends VLMRewardTrainer with Orthogonal Gradient Descent.

    After computing grad_new from the new-task batch, we also compute grad_ref
    from a reference-task batch. If the two gradients conflict (obtuse angle),
    we project grad_new onto the orthogonal complement of grad_ref.

    Memory-efficient: operates per-parameter instead of concatenating all grads.
    """

    def __init__(self, ref_dataloader=None, *args, **kwargs):
        _processing_class = kwargs.pop("processing_class", None)
        super().__init__(*args, **kwargs)
        if _processing_class is not None:
            self.processing_class = _processing_class
            if hasattr(self, "callback_handler") and self.callback_handler is not None:
                self.callback_handler.processing_class = _processing_class
        self.ref_dataloader = ref_dataloader
        self._ref_iter = None
        # Logging counters
        self._ogd_project_count = 0
        self._ogd_pass_count = 0
        self._ogd_total_count = 0

    def _get_ref_batch(self):
        """Get next batch from reference dataloader (infinite cycling)."""
        if self._ref_iter is None:
            self._ref_iter = iter(self.ref_dataloader)
        try:
            batch = next(self._ref_iter)
        except StopIteration:
            self._ref_iter = iter(self.ref_dataloader)
            batch = next(self._ref_iter)
        return batch

    def _get_trainable_params(self, model):
        """Return list of (name, param) for trainable params with gradients."""
        return [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        ]

    def _store_grads(self, model):
        """Clone current .grad for each trainable parameter. Returns dict name->grad."""
        grads = {}
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = p.grad.detach().clone()
        return grads

    def _compute_dot_and_norm(self, grad_a, grad_b):
        """
        Compute dot(grad_a, grad_b) and ||grad_b||^2 incrementally per-parameter.
        Both are dicts of name -> tensor.
        """
        dot = torch.tensor(0.0, device="cuda")
        norm_sq = torch.tensor(0.0, device="cuda")
        for name in grad_a:
            if name in grad_b:
                ga = grad_a[name].flatten().float()
                gb = grad_b[name].flatten().float()
                dot += torch.dot(ga, gb)
                norm_sq += torch.dot(gb, gb)
        return dot, norm_sq

    def _apply_projected_grad(self, model, grad_new, grad_ref, proj_coeff):
        """
        Write projected gradient into model .grad fields:
        grad_projected = grad_new - proj_coeff * grad_ref
        """
        for n, p in model.named_parameters():
            if p.requires_grad and n in grad_new:
                projected = grad_new[n] - proj_coeff * grad_ref[n] if n in grad_ref else grad_new[n]
                if p.grad is None:
                    p.grad = projected.clone()
                else:
                    p.grad.copy_(projected)

    def _apply_stored_grad(self, model, stored_grads):
        """Write stored gradients back into model .grad fields."""
        for n, p in model.named_parameters():
            if p.requires_grad and n in stored_grads:
                if p.grad is None:
                    p.grad = stored_grads[n].clone()
                else:
                    p.grad.copy_(stored_grads[n])

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with gradient orthogonal projection.

        1. Forward + backward on new data -> store grad_new per-param
        2. Forward + backward on reference data -> grad_ref per-param
        3. Compute dot(grad_new, grad_ref) and ||grad_ref||^2 incrementally
        4. If dot <= 0: project grad_new, write back to .grad
        5. Otherwise: restore grad_new to .grad
        """
        model.train()

        # -- Step 1: Compute grad_new from new-task data --
        inputs = self._prepare_inputs(inputs)
        loss_new = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss_new = loss_new / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss_new)

        # Store grad_new (clone per-param)
        grad_new = self._store_grads(model)

        # -- Step 2: Compute grad_ref from reference data --
        if self.ref_dataloader is not None and len(grad_new) > 0:
            # Zero grads before reference forward
            model.zero_grad()

            ref_batch = self._get_ref_batch()
            ref_batch = self._prepare_inputs(ref_batch)
            loss_ref = self.compute_loss(model, ref_batch)

            if self.args.gradient_accumulation_steps > 1:
                loss_ref = loss_ref / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss_ref)

            grad_ref = self._store_grads(model)

            # -- Step 3: Gradient projection --
            if len(grad_ref) > 0:
                dot, ref_norm_sq = self._compute_dot_and_norm(grad_new, grad_ref)

                self._ogd_total_count += 1

                if dot.item() <= 0 and ref_norm_sq.item() > 1e-12:
                    # Obtuse angle: project grad_new
                    # grad_projected = grad_new - (dot / ||grad_ref||^2) * grad_ref
                    proj_coeff = (dot / ref_norm_sq).item()
                    self._apply_projected_grad(model, grad_new, grad_ref, proj_coeff)
                    self._ogd_project_count += 1
                else:
                    # Acute angle: restore grad_new as-is
                    self._apply_stored_grad(model, grad_new)
                    self._ogd_pass_count += 1
            else:
                # No reference gradient, restore grad_new
                self._apply_stored_grad(model, grad_new)

            # Free reference grads
            del grad_ref
        # else: no ref_dataloader, grad_new is already in .grad from step 1

        # Free stored grad_new
        del grad_new

        # Log OGD stats periodically
        if self._ogd_total_count > 0 and self._ogd_total_count % 50 == 0:
            proj_ratio = self._ogd_project_count / self._ogd_total_count
            print(
                f"[OGD] step={self.state.global_step} | "
                f"project_ratio={proj_ratio:.3f} "
                f"({self._ogd_project_count}/{self._ogd_total_count})",
                flush=True,
            )

        return loss_new.detach()


# --- Main training function ---

def train(config, local_rank=0, debug=False):

    maybe_init_dist()

    ## ===> Step 1: Parse arguments
    (data_config, training_args, model_config, peft_lora_config), config_path = (
        parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig), config, is_train=True
        )
    )
    run_name = config.split("/")[-1].split(".")[0]
    timestamp = get_run_timestamp()
    training_args.output_dir = os.path.join(
        training_args.output_dir, f"{run_name}_{timestamp}"
    )
    training_args.logging_dir = training_args.output_dir
    training_args.save_only_model = True
    if training_args.report_to is None or training_args.report_to == "none":
        training_args.report_to = ["tensorboard"]
    elif isinstance(training_args.report_to, str):
        training_args.report_to = [training_args.report_to]

    assert not (
        peft_lora_config.lora_enable and model_config.freeze_llm
    ), "When using LoRA, the LLM should not be frozen."
    if not peft_lora_config.lora_enable:
        assert not peft_lora_config.vision_lora
    else:
        if peft_lora_config.lora_namespan_exclude is None:
            peft_lora_config.lora_namespan_exclude = []
        if not peft_lora_config.vision_lora:
            peft_lora_config.lora_namespan_exclude += ["visual"]

    ## ===> Step 2: Load model and configure
    model, processor, peft_config = create_model_and_processor(
        model_config=model_config,
        peft_lora_config=peft_lora_config,
        training_args=training_args,
    )

    if training_args.load_from_pretrained is not None:
        model, checkpoint_step = load_model_from_checkpoint(
            model,
            training_args.load_from_pretrained,
            training_args.load_from_pretrained_step,
        )
    model.train()

    if peft_lora_config.lora_enable:
        model_to_configure = model.model
    else:
        model_to_configure = model
        # Qwen3-VL: the LLM lives under model.model.language_model
        # Qwen2-VL: the LLM lives directly under model.model
        inner_model = model_to_configure.model
        if hasattr(inner_model, 'language_model'):
            llm_module = inner_model.language_model
        else:
            llm_module = inner_model
        set_requires_grad(
            llm_module.parameters(), not model_config.freeze_llm
        )
        set_requires_grad(llm_module.embed_tokens.parameters(), False)
    if not peft_lora_config.vision_lora:
        set_requires_grad(
            model_to_configure.visual.parameters(), not model_config.freeze_vision_tower
        )
        set_requires_grad(
            model_to_configure.visual.merger.parameters(), model_config.tune_merger
        )

    if model_config.trainable_visual_layers:
        assert model_config.trainable_visual_layers <= len(model_to_configure.visual.blocks)
        freeze_layer_num = (
            len(model_to_configure.visual.blocks) - model_config.trainable_visual_layers
            if model_config.trainable_visual_layers > 0
            else 0
        )
        for index, layer in enumerate(model_to_configure.visual.blocks):
            if index < freeze_layer_num:
                set_requires_grad(layer.parameters(), False)
            else:
                set_requires_grad(layer.parameters(), True)

    set_requires_grad(model_to_configure.rm_head.parameters(), True)

    ## ===> Step 3: Load datasets
    # Read the raw YAML to support the open-source CSV entry points (train_json / ref_json / test_json_list / data_root)
    import yaml
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    data_root = raw_config.get("data_root", None)
    train_json = raw_config.get("train_json", None)
    ref_json = raw_config.get("ref_json", None)
    test_json_list = raw_config.get("test_json_list", None)

    # New task dataset (Stage 1 labeled training data: aes/pf train split)
    if train_json:
        train_dataset = PairwiseOriginalDataset(
            data_json_list=([train_json] if isinstance(train_json, str) else train_json),
            soft_label=data_config.soft_label,
            confidence_threshold=data_config.confidence_threshold,
            data_root=data_root,
        )
    else:
        train_dataset = PairwiseOriginalDataset(
            data_config.train_json_list,
            data_config.soft_label,
            data_config.confidence_threshold,
        )

    # Reference dataset (HPDv3) - OGD anti-forgetting reference set
    ref_confidence_threshold = raw_config.get("ref_confidence_threshold", None)
    ref_batch_size = raw_config.get("ref_batch_size", 2)
    ref_json_list = raw_config.get("ref_json_list", [])

    if ref_json:
        ref_dataset = PairwiseOriginalDataset(
            data_json_list=([ref_json] if isinstance(ref_json, str) else ref_json),
            soft_label=data_config.soft_label,
            confidence_threshold=ref_confidence_threshold,
            data_root=data_root,
        )
        print(f"===> Loaded {len(ref_dataset)} reference samples for OGD.")
    elif ref_json_list:
        ref_dataset = PairwiseOriginalDataset(
            ref_json_list,
            data_config.soft_label,
            ref_confidence_threshold,
        )
        print(f"===> Loaded {len(ref_dataset)} reference samples for OGD.")
    else:
        ref_dataset = None
        print("===> WARNING: No ref data provided, OGD will be disabled (plain continue learning).")

    test_set_dict = {}
    if test_json_list:
        for test_name, test_json in test_json_list:
            test_set_dict[test_name] = PairwiseOriginalDataset(
                data_json_list=[test_json],
                soft_label=data_config.soft_label,
                confidence_threshold=data_config.confidence_threshold,
                data_root=data_root,
            )
    else:
        for item in data_config.test_json_list:
            test_set_dict[item[0]] = PairwiseOriginalDataset(
                item[1],
                data_config.soft_label,
                data_config.confidence_threshold,
            )

    print(f"===> Selected {len(train_dataset)} samples for training.")
    for key, value in test_set_dict.items():
        print(f"===> Selected {len(value)} samples for {key} testing.")

    num_gpu = int(os.environ.get("WORLD_SIZE", 1))
    data_collator = QWen2VLDataCollator(
        processor,
        max_pixels=data_config.max_pixels,
        min_pixels=data_config.min_pixels,
        with_instruction=data_config.with_instruction,
        use_special_tokens=model_config.use_special_tokens,
    )
    compute_metrics = partial(compute_multi_attr_accuracy)

    actual_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * num_gpu
    )
    total_steps = (
        training_args.num_train_epochs * len(train_dataset) // actual_batch_size
    )
    if training_args.save_epochs is not None:
        training_args.save_steps = round(
            training_args.save_epochs * len(train_dataset) / actual_batch_size
        )
    if training_args.eval_epochs is not None:
        training_args.eval_steps = round(
            training_args.eval_epochs * len(train_dataset) / actual_batch_size
        )
    if training_args.logging_epochs is not None:
        training_args.logging_steps = round(
            training_args.logging_epochs * len(train_dataset) / actual_batch_size
        )

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        print(f"===> Using {num_gpu} GPUs.")
        print(f"===> Total Batch Size: {actual_batch_size}")
        print(f"===> Training Epochs: {training_args.num_train_epochs}")
        print(f"===> Total Steps: {total_steps}")
        print(f"===> Save Steps: {training_args.save_steps}")
        print(f"===> Eval Steps: {training_args.eval_steps}")
        print(f"===> Logging Steps: {training_args.logging_steps}")
        print(f"===> OGD Mode: {'ENABLED' if ref_dataset else 'DISABLED'}")

    ## ===> Step 4: Build reference dataloader
    ref_dataloader = None
    if ref_dataset is not None:
        ref_sampler = None
        if dist.is_available() and dist.is_initialized():
            ref_sampler = DistributedSampler(
                ref_dataset, shuffle=True,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
            )
        ref_dataloader = DataLoader(
            ref_dataset,
            batch_size=ref_batch_size,
            sampler=ref_sampler,
            shuffle=(ref_sampler is None),
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
        )

    ## ===> Step 5: Save configs
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        save_configs_to_json(data_config, training_args, model_config, peft_lora_config)

    ## ===> Step 6: Start Training with OGD!
    special_token_ids = model.special_token_ids
    callbacks = []
    if special_token_ids is not None:
        callbacks.append(PartialEmbeddingUpdateCallback(special_token_ids))
    callbacks.append(WandbTrainLossCallback())

    trainer = OGDRewardTrainer(
        ref_dataloader=ref_dataloader,
        model=model,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(test_set_dict if training_args.conduct_eval else None),
        peft_config=peft_config,
        callbacks=callbacks,
        loss_type=model_config.loss_type,
        loss_hyperparameters=model_config.loss_hyperparameters,
        processing_class=processor.tokenizer,
        tied_threshold=data_config.tied_threshold,
        visualization_steps=training_args.visualization_steps,
        max_viz_samples=training_args.max_viz_samples,
    )
    trainer.train()

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        model_state_dict = model.state_dict()
        torch.save(
            model_state_dict, os.path.join(training_args.output_dir, "final_model.pth")
        )
        model.config.save_pretrained(training_args.output_dir)
        print(f"===> OGD training complete. Model saved to {training_args.output_dir}")
        print(
            f"===> OGD stats: projected {trainer._ogd_project_count}/{trainer._ogd_total_count} steps "
            f"({trainer._ogd_project_count / max(trainer._ogd_total_count, 1) * 100:.1f}%)"
        )


if __name__ == "__main__":
    fire.Fire(train)
