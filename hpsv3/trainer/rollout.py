"""
HPSv3 RM Training with Diffusion Rollout Data

Training strategy:
- Use the diffusion model's rollout outputs as unlabeled data
- For the multiple rollout samples of each prompt, score them with the old RM model
- Select training samples according to the score standard deviation (std):
  - max_std: select the samples with the largest std (active learning)
  - min_std: select the samples with the smallest std (curriculum learning)
"""

import json
import os
import math
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from types import SimpleNamespace

import csv
import fire
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

from functools import partial
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
from transformers import AutoProcessor, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import get_kbit_device_map, get_quantization_config
from hpsv3.model.differentiable_image_processor import Qwen2VLImageProcessor

try:
    import wandb
except ImportError:
    wandb = None

# Import shared utilities
from hpsv3.trainer.common import (
    maybe_init_dist,
    get_run_timestamp,
    create_model_and_processor,
    set_requires_grad,
    WandbTrainLossCallback,
)


class CsvLoggingCallback(TrainerCallback):
    """Append every on_log event to <output_dir>/training_log.csv (rank-0 only)."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        # Merge step into row
        row = {"step": state.global_step, **{k: v for k, v in logs.items()}}
        write_header = not self._header_written and not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ============================================================
# Rollout Dataset
# ============================================================





# ============================================================
# Rollout Trainer
# ============================================================

class RolloutRewardTrainer(VLMRewardTrainer):
    """
    Unsupervised RM trainer using rollout images (no human labels).

    Training format: each batch item = one prompt with ALL k rollout images.
    The collator flattens them to [B*k, ...] for a single forward pass.

    Loss = +/-std_mean + l2_coef * mean(r^2) + kl_coef * mean((r - r_ref)^2)
      max_std_unsup: -std -> more discriminative RM
      min_std_unsup: +std -> more consistent RM
      l2_coef: bounds reward magnitude (prevents +/-inf collapse)
      kl_coef: keeps model near reference (prevents arbitrary score inversion)

    Eval format: pairwise (PairwiseOriginalDataset) -> falls back to super().
    """

    def __init__(self, *args, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self._ref_on_device = False   # lazy device placement
        self._last_vis_step = -1      # dedup: only log once per optimizer step

    def create_optimizer(self):
        """Override to handle special_token_ids duplicate param group issue."""
        backup = None
        if self.args.special_token_lr is None and hasattr(self.model, 'special_token_ids'):
            backup = self.model.special_token_ids
            self.model.special_token_ids = []
        try:
            result = super().create_optimizer()
        finally:
            if backup is not None:
                self.model.special_token_ids = backup
        return result

    def _save_checkpoint(self, model, trial, **kwargs):
        """Override to handle API changes across transformers versions."""
        try:
            super()._save_checkpoint(model, trial, **kwargs)
        except TypeError:
            super(VLMRewardTrainer, self)._save_checkpoint(model, trial)

    # ------------------------------------------------------------------ #
    # Core loss: std-based unsupervised objective                         #
    # ------------------------------------------------------------------ #

    def _compute_std_loss(self, model, inputs, return_outputs=False):
        """
        One forward pass over all B*k images, then compute std per prompt.

        Loss = +/-std_mean + l2_coef * mean(r^2) + kl_coef * mean((r - r_ref)^2)

        l2_coef: bounds reward magnitude, prevents +/-inf drift (replaces hard clamp
                 which kills gradients at boundary)
        kl_coef: KL constraint via reference model -- forces score changes to be
                 anchored to the reference model's ordering, preventing arbitrary
                 sign flips or collapse to constant outputs
        """
        rewards = model(return_dict=True, **inputs["batch_all"])["logits"]  # [B*k, out_dim]
        r = rewards[:, 0]   # raw rewards, no clamp

        stds = []
        groups = []
        start = 0
        for k in inputs["k_per_prompt"]:
            groups.append((start, k))
            stds.append(r[start : start + k].std(unbiased=False))
            start += k
        std_batch = torch.stack(stds)   # [B]

        # L2 reg: bound reward magnitude with gradient flow
        l2_coef = getattr(self.args, "reward_l2_coef", 0.01)
        l2_reg = l2_coef * r.pow(2).mean()

        if self.loss_type == "max_std_unsup":
            loss = -std_batch.mean() + l2_reg
        else:   # min_std_unsup
            loss = std_batch.mean() + l2_reg

        # KL constraint: penalise deviation from frozen reference model
        kl_coef = getattr(self.args, "kl_coef", 0.0)
        kl_loss = torch.tensor(0.0, device=r.device)
        if kl_coef > 0.0 and self.ref_model is not None:
            if not self._ref_on_device:
                self.ref_model = self.ref_model.to(r.device)
                self._ref_on_device = True
            with torch.no_grad():
                r_ref = self.ref_model(
                    return_dict=True, **inputs["batch_all"]
                )["logits"][:, 0]
            kl_loss = kl_coef * (r - r_ref).pow(2).mean()
            loss = loss + kl_loss

        if wandb is not None and wandb.run is not None:
            step = self.state.global_step if hasattr(self, "state") else 0
            r_det = r.detach()
            log = {
                "rollout/reward_std":  std_batch.detach().mean().item(),
                "rollout/reward_mean": r_det.mean().item(),
                "rollout/reward_max":  r_det.max().item(),
                "rollout/reward_min":  r_det.min().item(),
                "rollout/l2_reg":      l2_reg.detach().item(),
            }
            if kl_coef > 0.0:
                log["rollout/kl_loss"] = kl_loss.detach().item()
            wandb.log(log, step=step)

            # Vis every vis_steps optimizer steps (use global_step, dedup per step)
            vis_steps = getattr(self.args, "vis_steps", 100)
            if step % vis_steps == 0 and step != self._last_vis_step:
                self._last_vis_step = step
                has_paths = "image_paths" in inputs
                print(f"[vis] step={step} vis_steps={vis_steps} has_image_paths={has_paths}")
                if has_paths:
                    self._log_wandb_images(inputs, r_det, groups, stds, step)

        if return_outputs:
            return loss, {"rewards_all": rewards}
        return loss

    def _log_wandb_images(self, inputs, r_det, groups, stds, step):
        """Log one prompt's images + scores to wandb for visual inspection."""
        try:
            # Pick the prompt with highest std in this batch (most informative)
            best_idx = int(torch.stack(stds).detach().argmax().item())
            start, k = groups[best_idx]
            scores = r_det[start : start + k].cpu().tolist()
            paths = inputs["image_paths"][best_idx]
            prompt = inputs["prompts"][best_idx] if "prompts" in inputs else ""
            std_val = stds[best_idx].item()
            print(f"[vis] logging {k} images for prompt: {prompt[:60]}  scores={[f'{s:.2f}' for s in scores]}")

            wandb_images = [
                wandb.Image(Image.open(p).convert("RGB"), caption=f"score={s:.3f}")
                for p, s in zip(paths, scores)
            ]
            # Use commit=False to avoid step conflict with the metrics log above
            wandb.log({
                "rollout/vis_images": wandb_images,
                "rollout/vis_std":    std_val,
                "rollout/vis_prompt": wandb.Html(f"<p>{prompt[:200]}</p>"),
            }, step=step, commit=False)
        except Exception as e:
            import traceback
            print(f"[wandb vis] FAILED at step {step}: {e}")
            traceback.print_exc()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Dispatch to std loss (train) or pairwise loss (eval)."""
        if "batch_all" in inputs:
            return self._compute_std_loss(model, inputs, return_outputs)
        # Pairwise format during eval -- no backward, so dual-forward is safe
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to fix accuracy computation for output_dim=2 models.

        Parent's prediction_step does:
          torch.cat([rewards_A [B,2], rewards_B [B,2]], dim=1) -> [B,4]
        Then compute_multi_attr_accuracy reads [:, 0] as chosen and [:, 1] as
        rejected -- but [:, 1] is rewards_A's log_sigma, not rewards_B's mean!

        Fix: always extract only dim-0 (mean reward) from each tensor -> [B,2].
        """
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(
                getattr(self.model, "config", None), "keys_to_ignore_at_inference", []
            ) or []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()

        if "rewards_A" in logits_dict and "rewards_B" in logits_dict:
            # Pairwise eval: extract mean-reward dim from both -> [B, 2]
            r_A = logits_dict["rewards_A"][:, [0]]   # mean reward of A (=path1=chosen)
            r_B = logits_dict["rewards_B"][:, [0]]   # mean reward of B (=path2=rejected)
            logits = torch.cat([r_A, r_B], dim=1).detach()
            # Diagnostic: print sample rewards to verify ordering is correct
            step = self.state.global_step if hasattr(self, "state") else -1
            if step % 500 == 0:
                ra_sample = r_A[:4, 0].cpu().tolist()
                rb_sample = r_B[:4, 0].cpu().tolist()
                acc_sample = sum(a > b for a, b in zip(ra_sample, rb_sample)) / max(len(ra_sample), 1)
                print(f"[eval diag] step={step}  r_A(chosen)[:4]={[f'{v:.3f}' for v in ra_sample]}"
                      f"  r_B(rejected)[:4]={[f'{v:.3f}' for v in rb_sample]}"
                      f"  local_acc={acc_sample:.3f}")
        else:
            # Rollout format during train -- return dummy (won't reach here in eval)
            vals = list(logits_dict.values())
            logits = torch.cat([v[:, [0]] for v in vals], dim=1).detach()

        labels = torch.ones((logits.shape[0], 1), device=logits.device)
        return loss, logits, labels


# ============================================================
# Config Loading
# ============================================================

def load_config(config_path: str):
    """Load config from YAML file as a simple namespace object."""
    import yaml
    from types import SimpleNamespace

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert to namespace for easy attribute access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    return dict_to_namespace(config_dict)


# ============================================================
# Main Training Function
# ============================================================



