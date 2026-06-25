# HPSv3++: Scaling Reward Models Across the Full Spectrum of Diffusion Model Capabilities

HPSv3++ is a **capability-aware and RL-iteration-aware** text-to-image (T2I) reward model. A Capability Encoder implicitly infers the generative ability of the model that produced an image, while the RL iteration step is supplied as an explicit condition. The two signals are jointly modulated through FiLM conditioning, so that a single reward model produces calibrated preference scores across the full spectrum of *generators of differing capability* and *different stages of RL optimization*.

<p align="center">
  <img src="assets/showcase.png" width="90%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.14657"><img src="https://img.shields.io/badge/arXiv-2606.14657-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/Junjun2333/HPSv3-PlusPlus"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-HPSv3++-yellow" alt="Model"></a>
  <a href="https://huggingface.co/datasets/Junjun2333/HPDv3-PlusPlus"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HPDv3++-yellow" alt="Dataset"></a>
  <a href="https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="Code"></a>
</p>

---

## Repository layout

```
.
|-- README.md
|-- requirements.txt
|-- train_stage1.sh / train_stage2.sh   # Two-stage training entry scripts
|-- eval.sh                             # Evaluate HPSv3++ on HPDv3++
|-- checkpoints/
|   |-- config.json
|   `-- hpsv3++.pth                      # Weights (17.6 GB), downloaded from HuggingFace
|-- datasets/                           # HPDv3++ dataset, downloaded from HuggingFace
|   |-- train/{train_aes,train_tf,stage1_labeled,stage1_ref,stage2_labeled,rollout,ogd_std}.json
|   |-- test/{test_aes,test_tf}.json
|   `-- images/                         # Unified image pool (deduplicated)
|-- hpsv3/
|   |-- train_stage1.py / train_stage2.py / inference.py
|   |-- config/{train_stage1,train_stage2}.yaml
|   |-- dataset/ model/ trainer/ utils/
`-- evaluate/evaluate.py                # HPDv3++ pairwise preference evaluation
```

---

## Dataset: HPDv3++

HPDv3++ is a two-axis preference dataset built on a frontier generator (Qwen-Image), annotated along **text-following** and **aesthetic quality**. It contains the full set of data used for HPSv3++ two-stage training and evaluation.

<p align="center"><img src="assets/data.png" width="92%"></p>

### Ready-to-use training and test splits

The following four files are self-contained preference data and can be used
**independently of the HPSv3++ model, training code, or evaluation code**:

| File | Pairs | Use |
|---|---|---|
| `train/train_aes.json` | 100,463 | Training -- aesthetic preference |
| `train/train_tf.json`  | 90,908  | Training -- text-following preference |
| `test/test_aes.json`   | 5,720   | Evaluation -- aesthetic |
| `test/test_tf.json`    | 4,465   | Evaluation -- text-following |

Each record is simply `{"path1": <preferred>, "path2": <non-preferred>, "prompt": <text>}`,
where `path1` is the human-preferred image and `path2` the less-preferred one (same
convention as HPSv3/HPDv3). All images they reference are in the split-tar image pool
hosted here (`images/qwen_image/...`); train and test are disjoint, including across the
aes/tf axes. Minimal example:

```python
import json
data = json.load(open("datasets/train/train_aes.json"))
for r in data:
    win  = "datasets/" + r["path1"]   # preferred image
    lose = "datasets/" + r["path2"]   # non-preferred image
    prompt = r["prompt"]
    # ... train your own reward model with (prompt, win > lose)
```

> The other JSON files (`stage1_labeled`, `stage1_ref`, `stage2_labeled`, `rollout`,
> `ogd_std`) are only needed to reproduce the HPSv3++ two-stage training pipeline below.
> In particular `stage1_ref.json` references the original HPDv3 images (download separately).

### Full schema

All data files are **JSON arrays of objects**, one record per element. Fields we did not annotate are left as `null`. Every image path (`image_path` / `path1` / `path2`) is a **relative** path of the form `images/...`, resolved against the dataset root `datasets/`.

**`train/{train_aes,train_tf}.json`, `test/{test_aes,test_tf}.json`** -- the ready-to-use preference pairs described above; each element holds only `path1, path2, prompt` (`path1` preferred). Train and test are disjoint, including across the aes/tf axes.

**`train/rollout.json`** -- Stage 2 unlabeled rollouts, long format, one image per element. Records are aggregated into image groups by `group_id` at training time; supervision comes from the within-group score std rather than human labels.

| Field | Meaning |
|---|---|
| `group_id` | Group identifier (same prompt + same tier + same `iter_step` form one group) |
| `source` | `capability` (multi-tier capability group) or `iteration` (RL iteration trajectory group) |
| `prompt` | Text prompt |
| `tier` | Generator tier |
| `iter_step` | RL iteration step (always 0 for the capability source) |
| `capability` | Continuous capability score mapped from `tier` |
| `iter_norm` | Normalized iteration value, `min(1, iter_step/1000)` |
| `level` | Discrete quality level (valid for the capability source) |
| `image_path` | Relative image path |
| `ogd_std` | Pre-computed per-group std (used by the std-guided loss) |

**`train/ogd_std.json`** -- each element is `{group_key, std}`, the pre-computed per-group std. The same values are already embedded in the `ogd_std` field of `rollout.json`; this file is an independent backup.

**`train/{stage1_labeled,stage2_labeled,stage1_ref}.json`** -- preference pairs:

| Field | Meaning |
|---|---|
| `path1` / `path2` | Preferred / non-preferred image (`path1` is better) |
| `prompt` | Text prompt |
| `choice_dist` / `confidence` / `model1` / `model2` | Annotation distribution / confidence / generator names (`null` when unannotated) |

---

## 1. Installation

```bash
conda create -n hpsv3pp python=3.10 -y
conda activate hpsv3pp
pip install -r requirements.txt
```

The `Qwen/Qwen3-VL-8B-Instruct` backbone is downloaded automatically from HuggingFace on first run. You may also pre-download it and set `model_name_or_path` in the config to a local path.

## 2. Download weights and dataset

The HPSv3++ weights and the HPDv3++ dataset are released on HuggingFace and are not stored in this repository. Download them and place them at the repository root.

- Model weights: [Junjun2333/HPSv3-PlusPlus](https://huggingface.co/Junjun2333/HPSv3-PlusPlus)
- Dataset (HPDv3++): [Junjun2333/HPDv3-PlusPlus](https://huggingface.co/datasets/Junjun2333/HPDv3-PlusPlus)

```bash
pip install -U "huggingface_hub[cli]"

# Model weights -> checkpoints/hpsv3++.pth
hf download Junjun2333/HPSv3-PlusPlus hpsv3++.pth --local-dir checkpoints

# Dataset (JSON annotations + split-tar image pool) -> datasets/
hf download Junjun2333/HPDv3-PlusPlus --repo-type dataset --local-dir datasets
cd datasets && cat images.tar.part* | tar -xf - && cd ..   # -> datasets/images/{qwen_image,rollout,thumbs}
```

The image pool we host contains our own generated images (`qwen_image`, `rollout`).
The `stage1_ref.json` reference pairs use the **original HPDv3 images**, which we do
not re-host; if you want to reproduce Stage 1 with them, download HPDv3 separately and
place its images under `datasets/images/hpdv3/`:

```bash
hf download MizzenAI/HPDv3 --repo-type dataset --include "images.tar.gz.*" --local-dir hpdv3_src
cat hpdv3_src/images.tar.gz.* | gunzip | tar -xv   # then place the images under datasets/images/hpdv3/
```

## 3. Training (two stages, run separately)

Both stages read the JSON files under `datasets/` directly, with no extra preprocessing. The default is single-node 8 GPUs; override with `NPROC=4 bash ...`. The released `hpsv3++.pth` corresponds to Stage 1 for 1 epoch and Stage 2 for 2 epochs (already set in the yaml files).

```bash
bash train_stage1.sh    # Stage 1: OGD continual learning
bash train_stage2.sh    # Stage 2: semi-supervised adaptive training
```

- **Stage 1** initialization: set `load_from_pretrained` in `hpsv3/config/train_stage1.yaml` to the HPSv3 8B Qwen3-VL aesthetic reward model trained on HPDv3 (Stage 0); you need to provide this checkpoint yourself.
- **Stage 2** initialization: set `load_from_pretrained` in `hpsv3/config/train_stage2.yaml` to the Stage 1 checkpoint. Leave it as `null` if you only run Stage 2 or use the released weights for inference.

## 4. Inference and evaluation

Evaluate the released checkpoint on the HPDv3++ test sets:

```bash
bash eval.sh    # Evaluate checkpoints/hpsv3++.pth on aes + tf
```

Or call the evaluator directly:

```bash
python evaluate/evaluate.py \
    --test_json datasets/test/test_aes.json \
    --config_path hpsv3/config/train_stage2.yaml \
    --checkpoint_path checkpoints/hpsv3++.pth \
    --img_root datasets --mode pair --batch_size 4 --num_processes 8
```

The evaluator reports pairwise preference accuracy: a pair is correct when the preferred image (`path1`) receives the higher reward.

### Using HPSv3++ as a reward model

For programmatic scoring (e.g. as the reward in T2I RL fine-tuning or for ranking generations), use `hpsv3/inference.py`:

```python
from hpsv3.inference import HPSv3RewardInferencer

scorer = HPSv3RewardInferencer(
    config_path="hpsv3/config/train_stage2.yaml",
    checkpoint_path="checkpoints/hpsv3++.pth",
)
rewards = scorer.reward(
    prompts=["a corgi running on the beach at sunset"],
    image_paths=["example.png"],
)
score = rewards[0][0].item()   # index [i][0] is the mean (mu), the final scalar reward
```

The call above uses no extra arguments, because HPSv3++ handles its two conditions as follows:

- **Model capability** is judged **internally**: the Capability Encoder infers it from the image itself, so you never pass it in.
- **RL iteration** is an explicit condition with a **default of `0.0`** (the pre-RL setting). The `reward()` method takes an optional `iter_step` argument, a normalized scalar in `[0, 1]` (training uses `iter_norm = min(1, raw_step / 1000)` over RL steps 0--1000). Leaving it at the default is correct for ordinary scoring.

**Recommended settings:**

- **General reward / preference scoring and ranking** -- use the default `iter_step=0.0` (the pre-RL setting). Capability is handled automatically, so a single call is all you need.
- **As the reward inside T2I RL fine-tuning** -- following the setting used in our paper, ramp the iteration condition **linearly from 0.3 to 1.0** over the course of RL training (rather than tying it to the raw step count), so the reward stays calibrated as the policy improves. For example, at progress `p = current_step / total_steps`, pass `iter_step = 0.3 + 0.7 * p`. `iter_step` accepts a scalar (shared across the batch) or a per-sample list / 1-D tensor of length `B`.
- Use the **mean** output (`rewards[i][0]`, i.e. mu) as the scalar reward; the second channel is an uncertainty estimate (sigma) and is not used for ranking.

---

## Method

<p align="center"><img src="assets/method.png" width="95%"></p>

**Stage 1** performs continual learning via Orthogonal Gradient Descent (OGD), extending the reward model to frontier generators without catastrophic forgetting. **Stage 2** is semi-supervised adaptive training that conditions the reward on model capability and RL iteration step through FiLM, supervised by labeled pairs and the within-group std of unlabeled rollouts.

---

## TODO

- [ ] Release the T2I RL fine-tuning code that uses HPSv3++ as the reward model (Flow-GRPO integration, with the 0.3 -> 1.0 iteration-condition schedule).

---

## Citation

```bibtex
@article{liu2026hpsv3pp,
  title   = {HPSv3++: Scaling Reward Models Across the Full Spectrum of Diffusion Model Capabilities},
  author  = {Liu, Yijun and Huang, Jie and Xue, Zeyue and Li, Yuming and He, Ruizhe and Li, Haoran and Ge, Shijia and Fu, Siming},
  journal = {arXiv preprint arXiv:2606.14657},
  year    = {2026}
}
```
