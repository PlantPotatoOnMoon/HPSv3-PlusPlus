import pdb
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import torch
from hpsv3.dataset.utils import process_vision_info
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 
Textual prompt - {text_prompt}


"""


# INSTRUCTION = """
# You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

# **Text Alignment:**  
# Assess how well the image matches the textual prompt across the following sub-dimensions:
# - **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
# - **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
# - **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
# - **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
# - **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
# Textual prompt - {text_prompt}


# """

INSTRUCTION_debug = """
{text_prompt}
"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""


class QWen2VLDataCollator:
    def __init__(
        self,
        processor,
        with_instruction=True,
        max_pixels=256 * 28 * 28,  # Default max pixels
        min_pixels=256 * 28 * 28,  # Default min pixels
        use_special_tokens=True,
    ):
        self.processor = processor
        self.with_instruction = with_instruction
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.use_special_tokens = use_special_tokens

    def _clean_message(
        self,
        texts,
        images,
        max_pixels=256 * 28 * 28,
        min_pixels=256 * 28 * 28,
        with_instruction=True,
        use_special_tokens=True,
    ):
        """
        remove unnecessary keys from message(very very necessary)
        """
        message_list = []
        for text, image in zip(texts, images):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        return message_list

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(
            attention_mask, padding, "constant", 0
        )

        return sequences_padded, attention_mask_padded

    def __call__(self, inputs, with_instruction=True):
        """
        Preprocess inputs to token sequences and return a batch.

        Supports three formats:
        - Pairwise (PairwiseOriginalDataset): each item has image_1/image_2 keys.
        - Rollout (RolloutPromptDataset): each item has an 'images' list (k images).
          All k images are flattened into a single [B*k, ...] batch so the trainer
          can compute std of RM scores across k images in ONE forward pass.
        - Conditioned Rollout: 'images' list with optional 'per_image_texts'
          (text condition) or 'image_levels' (embedding condition).
        """
        if 'same_images' in inputs[0]:
            # DualGroupDataset: each item contains same_images and cross_images
            return self._collate_dual_group(inputs)
        if 'images' in inputs[0]:
            # Check for per_image_texts (STD dataset in text-condition mode)
            if 'per_image_texts' in inputs[0]:
                return self._collate_conditioned_rollout(inputs)
            return self._collate_rollout(inputs)
        return self._collate_pairwise(inputs)

    def _collate_rollout(self, inputs):
        """Collate rollout samples: flatten all k images per prompt into one batch."""
        all_messages = []
        k_per_prompt = []

        for item in inputs:
            images = item['images']
            prompt = item['prompt']
            k = len(images)
            k_per_prompt.append(k)

            messages = self._clean_message(
                [prompt] * k,
                images,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
                with_instruction=False,   # prompt already formatted in dataset
                use_special_tokens=self.use_special_tokens,
            )
            all_messages.extend(messages)

        image_inputs, _ = process_vision_info(all_messages)
        image_inputs = [np.array(img) / 255.0 for img in image_inputs]

        batch_all = self.processor(
            text=self.processor.apply_chat_template(
                all_messages, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": False},
        )

        result = {
            "batch_all": batch_all,
            "k_per_prompt": k_per_prompt,           # list[int], len=B
            "prompts": [item["prompt"] for item in inputs],  # list[str], for vis
        }
        # Flatten group ids: [B * k]
        group_ids = []
        for i, item in enumerate(inputs):
            group_ids.extend([i] * len(item["images"]))
        result["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
        if "rollout_std" in inputs[0]:
            result["rollout_std"] = torch.tensor(
                [item["rollout_std"] for item in inputs], dtype=torch.float32
            )
        if "image_paths" in inputs[0]:
            result["image_paths"] = [item["image_paths"] for item in inputs]  # list[list[str]]
        if "image_levels" in inputs[0]:
            # Flatten levels: [B * k]
            all_levels = []
            for item in inputs:
                all_levels.extend(item["image_levels"])
            result["level_ids"] = torch.tensor(all_levels, dtype=torch.long)
        if "iter_values" in inputs[0]:
            # Flatten iter_values: [B * k]
            all_iters = []
            for item in inputs:
                all_iters.extend(item["iter_values"])
            result["iter_values"] = torch.tensor(all_iters, dtype=torch.float32)
        if "cond_values" in inputs[0]:
            all_cond = []
            for item in inputs:
                all_cond.extend(item["cond_values"])
            result["cond_values"] = torch.tensor(all_cond, dtype=torch.float32)
        if "group_key" in inputs[0]:
            result["group_keys"] = [item["group_key"] for item in inputs]
        if "ogd_std" in inputs[0]:
            result["ogd_std"] = torch.tensor(
                [float(item.get("ogd_std", 0.0)) for item in inputs], dtype=torch.float32
            )
        if "is_same_tier" in inputs[0]:
            result["is_same_tier"] = torch.tensor(
                [item["is_same_tier"] for item in inputs], dtype=torch.bool
            )
        return result

    def _collate_conditioned_rollout(self, inputs):
        """Collate conditioned rollout: each image uses its own text (with condition description)."""
        all_messages = []
        k_per_prompt = []

        for item in inputs:
            images = item['images']
            per_image_texts = item['per_image_texts']  # per-image text with its own condition
            k = len(images)
            k_per_prompt.append(k)

            messages = self._clean_message(
                per_image_texts,
                images,
                max_pixels=self.max_pixels,
                min_pixels=self.min_pixels,
                with_instruction=False,
                use_special_tokens=self.use_special_tokens,
            )
            all_messages.extend(messages)

        image_inputs, _ = process_vision_info(all_messages)
        image_inputs = [np.array(img) / 255.0 for img in image_inputs]

        batch_all = self.processor(
            text=self.processor.apply_chat_template(
                all_messages, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": False},
        )

        result = {
            "batch_all": batch_all,
            "k_per_prompt": k_per_prompt,
            "prompts": [item["prompt"] for item in inputs],
        }
        group_ids = []
        for i, item in enumerate(inputs):
            group_ids.extend([i] * len(item["images"]))
        result["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
        if "image_paths" in inputs[0]:
            result["image_paths"] = [item["image_paths"] for item in inputs]
        if "image_levels" in inputs[0]:
            # Flatten levels: [B * k]
            all_levels = []
            for item in inputs:
                all_levels.extend(item["image_levels"])
            result["level_ids"] = torch.tensor(all_levels, dtype=torch.long)
        if "iter_values" in inputs[0]:
            all_iters = []
            for item in inputs:
                all_iters.extend(item["iter_values"])
            result["iter_values"] = torch.tensor(all_iters, dtype=torch.float32)
        if "cond_values" in inputs[0]:
            all_cond = []
            for item in inputs:
                all_cond.extend(item["cond_values"])
            result["cond_values"] = torch.tensor(all_cond, dtype=torch.float32)
        if "group_key" in inputs[0]:
            result["group_keys"] = [item["group_key"] for item in inputs]
        if "ogd_std" in inputs[0]:
            result["ogd_std"] = torch.tensor(
                [float(item.get("ogd_std", 0.0)) for item in inputs], dtype=torch.float32
            )
        return result

    def _collate_pairwise(self, inputs, with_instruction=True):
        """Original pairwise collation logic (image_1 vs image_2)."""
        images_1, images_2, texts_1, texts_2 = [], [], [], []

        for idx, batch in enumerate(inputs):
            texts_1.append(batch["text_1"])
            texts_2.append(batch["text_2"])
            images_1.append(batch["image_1"])
            images_2.append(batch["image_2"])

        messages_batch_1 = self._clean_message(
            texts_1,
            images_1,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            with_instruction=self.with_instruction,
            use_special_tokens=self.use_special_tokens,
        )
        messages_batch_2 = self._clean_message(
            texts_2,
            images_2,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            with_instruction=self.with_instruction,
            use_special_tokens=self.use_special_tokens,
        )
        image_inputs_1, _ = process_vision_info(messages_batch_1)
        image_inputs_2, _ = process_vision_info(messages_batch_2)
        image_inputs_1 = [
            np.array(image_inputs_1[i]) / 255.0 for i in range(len(image_inputs_1))
        ]
        image_inputs_2 = [
            np.array(image_inputs_2[i]) / 255.0 for i in range(len(image_inputs_2))
        ]
        do_rescale = False

        batch_1 = self.processor(
            text=self.processor.apply_chat_template(
                messages_batch_1, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs_1,
            videos=None,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": do_rescale},
        )
        batch_2 = self.processor(
            text=self.processor.apply_chat_template(
                messages_batch_2, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs_2,
            videos=None,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": do_rescale},
        )

        max_len = max(batch_1["input_ids"].shape[1], batch_2["input_ids"].shape[1])
        batch_1["input_ids"], batch_1["attention_mask"] = self._pad_sequence(
            batch_1["input_ids"], batch_1["attention_mask"], max_len, "right"
        )
        batch_2["input_ids"], batch_2["attention_mask"] = self._pad_sequence(
            batch_2["input_ids"], batch_2["attention_mask"], max_len, "right"
        )

        batch = {
            "batch_1": batch_1,
            "batch_2": batch_2,
            "choice_dist": torch.stack([batch["choice_dist"] for batch in inputs]),
            # Store original text prompts for visualization
            "text_1": texts_1,
            "text_2": texts_2,
            "image_1": image_inputs_1,
            "image_2": image_inputs_2,
        }

        # Conditioned pairwise: pass through level_ids
        if "level_1" in inputs[0]:
            batch["level_ids_1"] = torch.tensor(
                [item["level_1"] for item in inputs], dtype=torch.long
            )
            batch["level_ids_2"] = torch.tensor(
                [item["level_2"] for item in inputs], dtype=torch.long
            )

        # RL iteration values
        if "iter_value_1" in inputs[0]:
            batch["iter_values_1"] = torch.tensor(
                [item["iter_value_1"] for item in inputs], dtype=torch.float32
            )
            batch["iter_values_2"] = torch.tensor(
                [item["iter_value_2"] for item in inputs], dtype=torch.float32
            )
        if "cond_value_1" in inputs[0]:
            batch["cond_values_1"] = torch.tensor(
                [item["cond_value_1"] for item in inputs], dtype=torch.float32
            )
            batch["cond_values_2"] = torch.tensor(
                [item["cond_value_2"] for item in inputs], dtype=torch.float32
            )

        return batch

    def _collate_dual_group(self, inputs):
        """Collate DualGroupDataset: each item contains a same_images group and a cross_images group.

        Returns:
            batch_same: processor batch for same-tier images
            k_same: list[int] number of same images per prompt
            batch_cross: processor batch for cross-tier images
            k_cross: list[int] number of cross images per prompt
        """
        same_messages = []
        cross_messages = []
        k_same = []
        k_cross = []
        all_same_paths = []  # flattened list of same-image paths, used by v5 pseudo-label

        for item in inputs:
            prompt = item["prompt"]
            same_imgs = item["same_images"]
            cross_imgs = item["cross_images"]
            k_s = len(same_imgs)
            k_c = len(cross_imgs)
            k_same.append(k_s)
            # Collect same-image paths (if the item contains same_paths)
            if "same_paths" in item:
                all_same_paths.extend(item["same_paths"][:k_s])
            k_cross.append(k_c)

            same_messages.extend(
                self._clean_message(
                    [prompt] * k_s,
                    same_imgs,
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                    with_instruction=False,
                    use_special_tokens=self.use_special_tokens,
                )
            )
            cross_messages.extend(
                self._clean_message(
                    [prompt] * k_c,
                    cross_imgs,
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                    with_instruction=False,
                    use_special_tokens=self.use_special_tokens,
                )
            )

        def _build_batch(msgs):
            img_inputs, _ = process_vision_info(msgs)
            img_inputs = [np.array(img) / 255.0 for img in img_inputs]
            return self.processor(
                text=self.processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                ),
                images=img_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                images_kwargs={"do_rescale": False},
            )

        batch_same = _build_batch(same_messages)
        batch_cross = _build_batch(cross_messages)

        result = {
            "batch_same": batch_same,
            "k_same": k_same,
            "batch_cross": batch_cross,
            "k_cross": k_cross,
            "prompts": [item["prompt"] for item in inputs],
            "tiers": [item["tier"] for item in inputs],
        }
        if all_same_paths:
            result["same_paths"] = all_same_paths
        return result
