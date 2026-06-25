#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-source JSON data loading utilities.

Restore the released long-format JSON files (datasets/train/rollout.json,
*_labeled.json, stage1_ref.json, datasets/test/*.json) into the in-memory
structures expected by the training/evaluation code, and join the relative
image paths (images/...) into absolute paths rooted at data_root. This keeps
the construction and __getitem__ logic of the downstream datasets completely
unchanged, guaranteeing identical behavior to training on the original JSON
without any information loss.

JSON schema (all are arrays of objects, each element being one long-format row):
  rollout.json:
    {group_id, source, prompt, tier, iter_step, capability, iter_norm,
     level, image_path, ogd_std}
    - source == "capability": corresponds to the original stage2_metadata
      (iter_step is always 0)
    - source == "iteration":  corresponds to the original rl_iter_metadata
  *_labeled.json / *_ref.json / *_clean.json:
    {path1, path2, prompt[, choice_dist, confidence, model1, model2]}
    - path1/path2 are relative paths images/...
    - choice_dist is a list in string form (e.g. "[8, 1]") or null
    - confidence is a numeric string (e.g. "0.98") or null
"""
import os
import json
from collections import OrderedDict


def _abs(data_root, rel):
    """Join a relative path (images/...) into an absolute one; return as-is if already absolute."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(data_root, rel)


def load_rollout_json(json_path, data_root):
    """Read rollout.json and restore it into (stage2_metadata, rl_iter_metadata, ogd_std_map).

    The two returned lists match the original JSON structure:
      stage2_metadata: [{"prompt", "images": {tier: [abs_path,...]}, "levels": {tier: level}}, ...]
      rl_iter_metadata: [{"prompt", "rl_steps": {step_str: [abs_path,...]}, "tier"}, ...]
      ogd_std_map: {group_key: std}  group_key = f"{prompt_idx}_{tier}_{iter_step}"

    group_id has the form cap_<idx>_<tier>_0 / iter_<idx>_<tier>_<step>, where <idx>
    is the original prompt index, used to reconstruct group_key (aligned with the
    training code's prompt_to_idx).
    """
    # Aggregate by source
    # capability: prompt -> {tier: [abs_path]}, prompt -> {tier: level}
    cap_images = OrderedDict()   # prompt -> {tier: [paths]}
    cap_levels = {}              # prompt -> {tier: level}
    cap_order = []               # preserve the first-appearance order of prompts
    # iteration: prompt -> {step: [abs_path]}, prompt -> tier
    iter_steps = OrderedDict()   # prompt -> {step_str: [paths]}
    iter_tier = {}               # prompt -> tier
    iter_order = []
    ogd_std_map = {}

    with open(json_path, "r") as f:
        rows = json.load(f)
    for row in rows:
            prompt = row["prompt"]
            tier = row["tier"]
            ap = _abs(data_root, row["image_path"])
            std = row.get("ogd_std", "")
            if row["source"] == "capability":
                if prompt not in cap_images:
                    cap_images[prompt] = OrderedDict()
                    cap_levels[prompt] = {}
                    cap_order.append(prompt)
                cap_images[prompt].setdefault(tier, []).append(ap)
                lvl = row.get("level", "")
                if lvl != "":
                    cap_levels[prompt][tier] = int(float(lvl))
            else:  # iteration
                if prompt not in iter_steps:
                    iter_steps[prompt] = OrderedDict()
                    iter_order.append(prompt)
                    iter_tier[prompt] = tier
                step_str = str(int(float(row["iter_step"])))
                iter_steps[prompt].setdefault(step_str, []).append(ap)
            # ogd_std: group_key is reconstructed from prompt_idx (see below);
            # stash it by group_id for now
            if std not in ("", None):
                # group_id: cap_<idx>_<tier>_0 or iter_<idx>_<tier>_<step>
                gid = row["group_id"]
                parts = gid.split("_", 1)[1]  # strip the cap_/iter_ prefix
                # parts = "<idx>_<tier>_<step>" is exactly the group_key
                ogd_std_map[parts] = float(std)

    stage2_metadata = []
    for prompt in cap_order:
        stage2_metadata.append({
            "prompt": prompt,
            "images": {t: cap_images[prompt][t] for t in cap_images[prompt]},
            "levels": cap_levels[prompt],
        })
    rl_iter_metadata = []
    for prompt in iter_order:
        rl_iter_metadata.append({
            "prompt": prompt,
            "rl_steps": {s: iter_steps[prompt][s] for s in iter_steps[prompt]},
            "tier": iter_tier[prompt],
        })
    return stage2_metadata, rl_iter_metadata, ogd_std_map


def load_pairwise_json(json_path, data_root):
    """Read *_labeled.json / *_ref.json / *_clean.json and restore the sample
    list expected by PairwiseOriginalDataset: each sample is a dict with
    path1/path2 already joined into absolute paths."""
    samples = []
    with open(json_path, "r") as f:
        rows = json.load(f)
    for row in rows:
            s = {
                "path1": _abs(data_root, row["path1"]),
                "path2": _abs(data_root, row["path2"]),
                "prompt": row["prompt"],
            }
            cd = row.get("choice_dist", "")
            if cd not in ("", None):
                try:
                    s["choice_dist"] = json.loads(cd)
                except Exception:
                    s["choice_dist"] = None
            else:
                s["choice_dist"] = None
            conf = row.get("confidence", "")
            s["confidence"] = float(conf) if conf not in ("", None) else None
            if row.get("model1"):
                s["model1"] = row["model1"]
            if row.get("model2"):
                s["model2"] = row["model2"]
            samples.append(s)
    return samples
