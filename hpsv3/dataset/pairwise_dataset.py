import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
from tqdm import tqdm

# Model name -> level mapping (consistent with conditioned_rollout_dataset.py / eval_condition.py)
MODEL_TO_LEVEL = {
    "sd15": 0, "pixart": 0,
    "sdxl": 1, "kolors": 1, "cogview4": 1, "hunyuan": 1,
    "flux": 2, "sd3": 2, "infinity": 2, "qwen_image": 2, "real_images": 2,
}


class PairwiseOriginalDataset(Dataset):
    def __init__(
        self,
        json_list=None,
        soft_label=False,
        confidence_threshold=None,
        data_json_list=None,
        data_root=None,
    ):
        self.samples = []
        if data_json_list is not None:
            from hpsv3.dataset.json_loader import load_pairwise_json
            for json_file in data_json_list:
                self.samples.extend(load_pairwise_json(json_file, data_root or "."))
        else:
            for json_file in json_list:
                with open(json_file, "r") as f:
                    data = json.load(f)
                self.samples.extend(data)

        self.soft_label = soft_label
        self.confidence_threshold = confidence_threshold

        # trl >= 0.16 RewardTrainer checks column_names to decide whether to preprocess.
        # This custom Dataset ships its own data_collator; declaring input_ids_chosen
        # makes it skip preprocessing.
        self.column_names = ["input_ids_chosen", "input_ids_rejected"]

        if confidence_threshold is not None:
            new_samples = []
            for sample in tqdm(
                self.samples, desc="Filtering samples according to confidence threshold"
            ):
                if sample.get("confidence") is None or sample.get("confidence", float("inf")) >= confidence_threshold \
                 or sample.get("confidence")=="null":
                    new_samples.append(sample)
            self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            index = idx
            try:
                return self.get_single_item(index)
            except Exception as e:
                print(f"Error processing sample at index {idx}: {e}")
                import traceback
                traceback.print_exc()
                index = random.randint(0, len(self.samples) - 1)
                if index == idx:
                    continue
                idx = index

    def get_single_item(self, idx):
        sample = self.samples[idx]
        # Load image paths (already absolute, or data_root-relative paths joined by json_loader)
        image_1 = sample["path1"]
        image_2 = sample["path2"]
        assert os.path.exists(image_1) and os.path.exists(image_2), f'{image_1} or {image_2}'
        text_1 = sample["prompt"]
        text_2 = sample["prompt"]

        # Process Label
        if self.soft_label:
            choice_dist = sorted(sample["choice_dist"], reverse=True)
            assert (
                torch.sum(torch.tensor(choice_dist)) > 0
            ), "Choice distribution cannot be zero."
            label = torch.tensor(choice_dist[0]) / torch.sum(torch.tensor(choice_dist))
        else:
            label = torch.tensor(1).float()
        # Get the choice_dist value
        choice_dist = sample.get("choice_dist")

        # If the value is None or not a list, use a default
        if choice_dist is None or not isinstance(choice_dist, list):
            choice_dist = [1.0, 0.0]

        result = {
            "image_1": image_1,
            "image_2": image_2,
            "text_1": text_1,
            "text_2": text_2,
            "label": label,
            "confidence": sample.get("confidence", 1.0),
            "choice_dist": torch.tensor(choice_dist),
        }

        # If model info is present, map it to level (used by conditioned eval)
        model1 = sample.get("model1")
        model2 = sample.get("model2")
        if model1 and model2:
            level1 = MODEL_TO_LEVEL.get(model1, 1)  # default L1
            level2 = MODEL_TO_LEVEL.get(model2, 1)
            result["level_1"] = level1
            result["level_2"] = level2

        return result