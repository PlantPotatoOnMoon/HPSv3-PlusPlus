"""Reward Model factory function: returns the Qwen3-VL reward model classes based on model_name_or_path.

Usage:
    from hpsv3.model.reward_model_factory import get_reward_model_classes
    BaseRM, FiLMRM = get_reward_model_classes(model_name_or_path)
    model = BaseRM.from_pretrained(model_name_or_path, ...)
"""

from transformers import AutoConfig


def get_reward_model_classes(model_name_or_path: str):
    """Automatically return the matching (BaseRM, FiLMContinuousRM) class pair based on the model config.

    Args:
        model_name_or_path: HuggingFace model ID or local path.

    Returns:
        The (BaseRMClass, FiLMContinuousRMClass) tuple.

    Raises:
        ValueError: if the model type is not supported.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_vl":
        from hpsv3.model.qwen3vl_rm import (
            Qwen3VLRewardModelBT,
            Qwen3VLRewardModelFiLMContinuous,
        )

        return Qwen3VLRewardModelBT, Qwen3VLRewardModelFiLMContinuous

    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}' from {model_name_or_path}. "
            f"Only Qwen3-VL (model_type='qwen3_vl') is supported in this release."
        )
