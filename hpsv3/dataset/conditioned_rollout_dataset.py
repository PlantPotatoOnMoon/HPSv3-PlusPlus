"""
Stage 2 conditioned rollout datasets.

Two training modes are supported:
1. Ranking mode (weak supervision): build cross-tier pairs from stage2_metadata.json
   - (qwen_image, sd15) -> qwen_image should score higher
   - format compatible with PairwiseOriginalDataset
2. STD mode (unsupervised): all images of one prompt form a group
   - format compatible with RolloutPromptDataset

Two ways of injecting the condition are supported:
- "embedding": return level_id (int), handled by the model's level_embedding
- "text": append a quality description after the prompt text, no model change needed
"""

import json
import os
import random
from itertools import combinations
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Model name -> quality tier
MODEL_LEVELS = {
    "sd15": 0,
    "sdxl": 1,
    "cogview4": 1,
    "qwen_image": 2,
    "qwenimage": 2,
    "flux": 2,
}
# Model name -> continuous capability score
MODEL_CAPABILITIES = {
    "sd15": 0.0,
    "sdxl": 0.35,
    "cogview4": 0.6,
    "qwen_image": 0.8,
    "qwenimage": 0.8,
    "flux": 1.0,
}

# RL iteration-step normalization mapping
RL_STEP_TO_ITER = {
    0: 0.0,
    500: 0.25,
    1000: 0.5,
    2000: 1.0,
}


def _safe_rl_step_to_iter(step_key: str) -> float:
    """Convert an RL step key into a [0,1]-normalized iteration value."""
    try:
        step_int = int(step_key)
    except Exception:
        return 0.0
    return RL_STEP_TO_ITER.get(step_int, 0.0 if step_int <= 0 else min(1.0, step_int / 2000.0))


class MixedConditionSTDDataset(Dataset):
    column_names = ["input_ids", "attention_mask", "pixel_values"]  # trl compat

    def map(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self
    """ExpY: STD dataset mixing model capability + RL iteration.

    Each item corresponds to a group (same prompt, same tier, same iter_step):
      - stage2_metadata: (sd15/sdxl/qwen_image, iter=0)
      - rl_iter_metadata: (qwen_image, iter in {0,500,1000,2000})

    Returns cond_values=[capability, iter_norm], optionally with ogd_std/group_key.
    """

    def __init__(
        self,
        metadata_path: str = None,
        rl_iter_metadata_path: str = None,
        ogd_std_path: Optional[str] = None,
        tiers: Optional[list] = None,
        rl_steps: Optional[list] = None,
        skip_missing: bool = True,
        max_images_per_group: int = 6,
        val_size: int = 500,
        split: str = "train",
        rl_tier: Optional[str] = None,
        max_rl_step: Optional[int] = None,
        json_path: Optional[str] = None,
        data_root: Optional[str] = None,
    ):
        self.skip_missing = skip_missing
        self.tiers = tiers or ["sd15", "sdxl", "qwen_image"]
        self.max_images_per_group = max_images_per_group
        self.rl_steps = rl_steps  # None = all available
        self.rl_tier = rl_tier  # None = "qwen_image" (backward compat)
        self.max_rl_step = max_rl_step  # None = use RL_STEP_TO_ITER dict

        if json_path is not None:
            # Open-source JSON entry: restore metadata from the unified long-format
            # rollout.json (paths joined with data_root)
            from hpsv3.dataset.json_loader import load_rollout_json
            stage2_metadata, rl_iter_metadata, self.ogd_std_map = load_rollout_json(
                json_path, data_root or "."
            )
        else:
            with open(metadata_path, "r") as f:
                stage2_metadata = json.load(f)
            with open(rl_iter_metadata_path, "r") as f:
                rl_iter_metadata = json.load(f)

            self.ogd_std_map = {}
            if ogd_std_path and os.path.exists(ogd_std_path):
                with open(ogd_std_path, "r") as f:
                    self.ogd_std_map = json.load(f)

        # prompt -> idx: prefer stage2's order; new prompts in RL keep incrementing
        prompt_to_idx = {item["prompt"]: i for i, item in enumerate(stage2_metadata)}
        next_prompt_idx = len(prompt_to_idx)

        self.data = []
        self.data.extend(
            self._build_stage2_groups(stage2_metadata, prompt_to_idx)
        )
        self.data.extend(
            self._build_rl_groups(rl_iter_metadata, prompt_to_idx, next_prompt_idx)
        )

        if split == "val":
            # Stratified val split: sample proportionally from each tier
            from collections import defaultdict
            tier_groups = defaultdict(list)
            for g in self.data:
                tier_groups[g["tier"]].append(g)
            val_data = []
            n_tiers = max(len(tier_groups), 1)
            for tier, groups in tier_groups.items():
                n_take = max(1, val_size * len(groups) // len(self.data))
                val_data.extend(groups[-n_take:])
            # Shuffle to interleave tiers (so max_groups in eval sees all tiers)
            import random
            random.Random(42).shuffle(val_data)
            self.data = val_data[:val_size]
        elif val_size > 0:
            self.data = self.data[:-val_size]

        print(
            f"[MixedConditionSTDDataset] {len(self.data)} groups "
            f"(split={split}, tiers={self.tiers})"
        )

    def _build_stage2_groups(self, metadata: list, prompt_to_idx: Dict[str, int]) -> list:
        groups = []
        for item in metadata:
            prompt = item["prompt"]
            prompt_idx = prompt_to_idx[prompt]
            images_dict = item.get("images", {})
            levels = item.get("levels", {})
            for tier in self.tiers:
                if tier not in images_dict:
                    continue
                paths = []
                for p in images_dict[tier]:
                    if self.skip_missing and not os.path.exists(p):
                        continue
                    paths.append(p)
                if len(paths) < 2:
                    continue
                paths = paths[: self.max_images_per_group]
                cap = MODEL_CAPABILITIES.get(tier, 0.5)
                iter_norm = 0.0
                iter_step = 0
                group_key = f"{prompt_idx}_{tier}_{iter_step}"
                groups.append(
                    {
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "image_paths": paths,
                        "tier": tier,
                        "level": levels.get(tier, MODEL_LEVELS.get(tier, 1)),
                        "capability": cap,
                        "iter_norm": iter_norm,
                        "iter_step": iter_step,
                        "group_key": group_key,
                        "ogd_std": self.ogd_std_map.get(group_key),
                    }
                )
        return groups

    def _build_rl_groups(
        self,
        rl_iter_metadata: list,
        prompt_to_idx: Dict[str, int],
        next_prompt_idx: int,
    ) -> list:
        groups = []
        for item in rl_iter_metadata:
            prompt = item.get("prompt", "")
            if prompt not in prompt_to_idx:
                prompt_to_idx[prompt] = next_prompt_idx
                next_prompt_idx += 1
            prompt_idx = prompt_to_idx[prompt]

            rl_steps_dict = item.get("rl_steps", {})
            for step_key, step_paths in rl_steps_dict.items():
                if self.rl_steps is not None and int(step_key) not in self.rl_steps:
                    continue
                iter_step = int(step_key)
                if self.max_rl_step is not None and self.max_rl_step > 0:
                    iter_norm = min(1.0, iter_step / self.max_rl_step)
                else:
                    iter_norm = _safe_rl_step_to_iter(step_key)
                paths = []
                for p in step_paths:
                    if self.skip_missing and not os.path.exists(p):
                        continue
                    paths.append(p)
                if len(paths) < 2:
                    continue
                paths = paths[: self.max_images_per_group]
                tier = item.get("tier", self.rl_tier or "qwen_image")
                cap = MODEL_CAPABILITIES.get(tier, 1.0)
                group_key = f"{prompt_idx}_{tier}_{iter_step}"
                groups.append(
                    {
                        "prompt": prompt,
                        "prompt_idx": prompt_idx,
                        "image_paths": paths,
                        "tier": tier,
                        "level": MODEL_LEVELS[tier],
                        "capability": cap,
                        "iter_norm": iter_norm,
                        "iter_step": iter_step,
                        "group_key": group_key,
                        "ogd_std": self.ogd_std_map.get(group_key),
                    }
                )
        return groups

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_paths = item["image_paths"]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        k = len(images)
        cond = [item["capability"], item["iter_norm"]]

        result = {
            "prompt": item["prompt"],
            "images": images,
            "image_paths": image_paths,
            "image_levels": [item["level"]] * k,
            "iter_values": [item["iter_norm"]] * k,
            "cond_values": [cond] * k,
            "group_key": item["group_key"],
            "tier": item["tier"],
            "iter_step": item["iter_step"],
            "rollout_std": 0.0,
        }
        if item["ogd_std"] is not None:
            result["ogd_std"] = float(item["ogd_std"])
        return result


class MixedConditionRankingDataset(Dataset):
    """ExpY: ranking-pair dataset across capability/iteration."""

    def __init__(
        self,
        metadata_path: str = None,
        rl_iter_metadata_path: str = None,
        tiers: Optional[list] = None,
        pairs_per_prompt: int = 3,
        pair_mode: str = "capability_then_iter",
        skip_missing: bool = True,
        rl_tier: Optional[str] = None,
        max_rl_step: Optional[int] = None,
        json_path: Optional[str] = None,
        data_root: Optional[str] = None,
    ):
        self.skip_missing = skip_missing
        self.tiers = tiers or ["sd15", "sdxl", "qwen_image"]
        self.pairs_per_prompt = pairs_per_prompt
        self.pair_mode = pair_mode
        self.rl_tier = rl_tier
        self.max_rl_step = max_rl_step

        if json_path is not None:
            # Open-source JSON entry: restore metadata from the unified long-format
            # rollout.json (paths joined with data_root)
            from hpsv3.dataset.json_loader import load_rollout_json
            stage2_metadata, rl_iter_metadata, _ = load_rollout_json(json_path, data_root or ".")
        else:
            with open(metadata_path, "r") as f:
                stage2_metadata = json.load(f)
            with open(rl_iter_metadata_path, "r") as f:
                rl_iter_metadata = json.load(f)

        prompt_records = self._build_prompt_records(stage2_metadata, rl_iter_metadata)
        self.pairs = self._build_pairs(prompt_records)
        print(f"[MixedConditionRankingDataset] {len(self.pairs)} pairs")

    @staticmethod
    def _ranking_priority(rec: dict):
        """Ranking supervision proxy for ExpY.

        Higher capability is preferred; for same capability, lower RL iteration
        is treated as preferred (to avoid forcing collapsed late-iteration outputs
        as "better" by default).
        """
        return (float(rec["capability"]), -float(rec["iter_norm"]))

    def _choose_pair_order(self, a: dict, b: dict):
        """Return (chosen, rejected) for a pair, or None to skip this pair."""
        if self.pair_mode == "cross_model_only":
            # Keep only cross-model-capability pairs, avoiding interference from
            # same-tier / same-capability internal pairs.
            if a["tier"] == b["tier"]:
                return None
            if float(a["capability"]) == float(b["capability"]):
                return None
            return (a, b) if float(a["capability"]) > float(b["capability"]) else (b, a)

        if self.pair_mode == "capability_only":
            # Order by capability only; discard if capabilities are equal.
            if float(a["capability"]) == float(b["capability"]):
                return None
            return (a, b) if float(a["capability"]) > float(b["capability"]) else (b, a)

        # Default: capability first; for equal capability, lower iter is preferred.
        p_a = self._ranking_priority(a)
        p_b = self._ranking_priority(b)
        if p_a == p_b:
            return None
        return (a, b) if p_a > p_b else (b, a)

    def _build_prompt_records(self, stage2_metadata: list, rl_iter_metadata: list) -> Dict[str, list]:
        prompt_records: Dict[str, list] = {}

        for item in stage2_metadata:
            prompt = item["prompt"]
            images_dict = item.get("images", {})
            levels = item.get("levels", {})
            recs = prompt_records.setdefault(prompt, [])
            for tier in self.tiers:
                if tier not in images_dict:
                    continue
                level = levels.get(tier, MODEL_LEVELS.get(tier, 1))
                cap = MODEL_CAPABILITIES.get(tier, 0.5)
                for p in images_dict[tier]:
                    if self.skip_missing and not os.path.exists(p):
                        continue
                    recs.append(
                        {
                            "path": p,
                            "tier": tier,
                            "level": level,
                            "capability": cap,
                            "iter_norm": 0.0,
                            "iter_step": 0,
                        }
                    )

        for item in rl_iter_metadata:
            prompt = item.get("prompt", "")
            recs = prompt_records.setdefault(prompt, [])
            for step_key, paths in item.get("rl_steps", {}).items():
                iter_step = int(step_key)
                if self.max_rl_step is not None and self.max_rl_step > 0:
                    iter_norm = min(1.0, iter_step / self.max_rl_step)
                else:
                    iter_norm = _safe_rl_step_to_iter(step_key)
                _rl_tier = item.get("tier", self.rl_tier or "qwen_image")
                for p in paths:
                    if self.skip_missing and not os.path.exists(p):
                        continue
                    recs.append(
                        {
                            "path": p,
                            "tier": _rl_tier,
                            "level": MODEL_LEVELS.get(_rl_tier, MODEL_LEVELS["qwen_image"]),
                            "capability": MODEL_CAPABILITIES.get(_rl_tier, 1.0),
                            "iter_norm": iter_norm,
                            "iter_step": iter_step,
                        }
                    )

        return prompt_records

    def _build_pairs(self, prompt_records: Dict[str, list]) -> list:
        pairs = []
        for prompt, recs in prompt_records.items():
            if len(recs) < 2:
                continue
            if self.pairs_per_prompt > 0:
                prompt_pairs = self._sample_prompt_pairs(prompt, recs, self.pairs_per_prompt)
            else:
                prompt_pairs = []
                for a, b in combinations(recs, 2):
                    pair = self._choose_pair_order(a, b)
                    if pair is None:
                        continue
                    chosen, rejected = pair
                    prompt_pairs.append(self._format_pair(prompt, chosen, rejected))
            pairs.extend(prompt_pairs)
        return pairs

    def _format_pair(self, prompt: str, chosen: dict, rejected: dict) -> dict:
        return {
            "prompt": prompt,
            "path1": chosen["path"],
            "path2": rejected["path"],
            "level1": chosen["level"],
            "level2": rejected["level"],
            "iter_value_1": chosen["iter_norm"],
            "iter_value_2": rejected["iter_norm"],
            "cond_value_1": [chosen["capability"], chosen["iter_norm"]],
            "cond_value_2": [rejected["capability"], rejected["iter_norm"]],
        }

    def _sample_prompt_pairs(self, prompt: str, recs: list, max_pairs: int) -> list:
        prompt_pairs = []
        seen = set()
        attempts = 0
        max_attempts = max(max_pairs * 32, len(recs) * 4)

        while len(prompt_pairs) < max_pairs and attempts < max_attempts:
            a, b = random.sample(recs, 2)
            attempts += 1
            pair = self._choose_pair_order(a, b)
            if pair is None:
                continue
            chosen, rejected = pair
            key = (chosen["path"], rejected["path"])
            if key in seen:
                continue
            seen.add(key)
            prompt_pairs.append(self._format_pair(prompt, chosen, rejected))

        if len(prompt_pairs) >= max_pairs:
            return prompt_pairs

        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                pair = self._choose_pair_order(recs[i], recs[j])
                if pair is None:
                    continue
                chosen, rejected = pair
                key = (chosen["path"], rejected["path"])
                if key in seen:
                    continue
                seen.add(key)
                prompt_pairs.append(self._format_pair(prompt, chosen, rejected))
                if len(prompt_pairs) >= max_pairs:
                    return prompt_pairs
        return prompt_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return {
            "image_1": pair["path1"],
            "image_2": pair["path2"],
            "text_1": pair["prompt"],
            "text_2": pair["prompt"],
            "label": torch.tensor(1.0),
            "confidence": 1.0,
            "choice_dist": torch.tensor([1.0, 0.0]),
            "level_1": pair["level1"],
            "level_2": pair["level2"],
            "iter_value_1": pair["iter_value_1"],
            "iter_value_2": pair["iter_value_2"],
            "cond_value_1": pair["cond_value_1"],
            "cond_value_2": pair["cond_value_2"],
        }
