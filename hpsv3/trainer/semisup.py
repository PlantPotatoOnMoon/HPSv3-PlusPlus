"""
HPSv3 Semi-Supervised Stage 2 Training

Combines labeled data (pairwise uncertainty loss) and unlabeled data (max-std rollout loss),
continuing training from an OGD-trained checkpoint.

Three training strategies:
1. alternating_batch: each step randomly picks a labeled or unlabeled batch
2. mixed_loss: each step computes both losses and takes their weighted sum
3. epoch_alternating: odd epochs use labeled data, even epochs use unlabeled data
"""

import json
import os
import csv
import copy
import math
import random
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from functools import partial
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset

from hpsv3.model.qwen2vl_trainer import (
    Qwen2VLRewardModelBT,
    VLMRewardTrainer,
    compute_multi_attr_accuracy,
    PartialEmbeddingUpdateCallback,
)
from hpsv3.dataset.pairwise_dataset import PairwiseOriginalDataset
from hpsv3.dataset.data_collator_qwen import QWen2VLDataCollator
from hpsv3.utils.parser import ModelConfig, PEFTLoraConfig
from hpsv3.utils.training_utils import load_model_from_checkpoint
from hpsv3.trainer.common import (
    maybe_init_dist,
    get_run_timestamp,
    create_model_and_processor,
    set_requires_grad,
    WandbTrainLossCallback,
)
from hpsv3.trainer.rollout import (
    RolloutRewardTrainer,
    CsvLoggingCallback,
    load_config,
)
from transformers import TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================
# Infinite DataLoader Iterator
# ============================================================

class InfiniteDataLoaderIterator:
    """DataLoader iterator that samples in an infinite loop."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
        return batch


# ============================================================
# Epoch Alternating Callback
# ============================================================



# ============================================================
# Semi-Supervised Reward Trainer
# ============================================================



# ============================================================
# Eval Train Subset utility functions
# ============================================================

def create_eval_train_subset(
    labeled_json_list: list,
    eval_train_size: int,
    output_dir: str,
    is_main: bool = True,
) -> PairwiseOriginalDataset:
    """Draw a fixed subset from the labeled data to serve as the training-data eval set.

    Indices are saved to output_dir/eval_train_10k_indices.json,
    and reused if the file already exists. Only rank 0 reads/writes the file,
    then broadcasts to all ranks.
    """
    indices_path = os.path.join(output_dir, "eval_train_10k_indices.json")

    full_dataset = PairwiseOriginalDataset(json_list=labeled_json_list)

    rank = dist.get_rank() if dist.is_initialized() else 0
    indices = None

    # Only rank 0 handles file IO, to avoid multi-process races
    if rank == 0:
        if os.path.exists(indices_path):
            with open(indices_path, "r") as f:
                indices = json.load(f)
            print(f"[eval_train] reusing existing indices: {indices_path} ({len(indices)} entries)")
        else:
            total = len(full_dataset)
            sample_size = min(eval_train_size, total)
            rng = np.random.RandomState(42)
            indices = rng.choice(total, size=sample_size, replace=False).tolist()
            os.makedirs(output_dir, exist_ok=True)
            with open(indices_path, "w") as f:
                json.dump(indices, f)
            print(f"[eval_train] created new indices: {indices_path} ({len(indices)} entries)")

    # Broadcast indices to ensure all ranks are consistent
    if dist.is_initialized():
        obj_list = [indices]
        dist.broadcast_object_list(obj_list, src=0)
        indices = obj_list[0]

    subset = Subset(full_dataset, indices)
    return subset


# ============================================================
# Main Training Function
# ============================================================



