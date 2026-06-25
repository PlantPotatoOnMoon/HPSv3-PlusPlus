"""Qwen3-VL Reward Model classes.

Functionally identical to the Qwen2-VL version (qwen2vl_trainer.py), but inherits from
Qwen3VLForConditionalGeneration. Requires transformers >= 4.57.0.

Key architectural differences (relative to Qwen2-VL):
- Qwen3-VL's self.model() handles vision encoding internally (no need to call self.visual manually)
- forward does not accept rope_deltas/use_cache/output_attentions/output_hidden_states
- hidden_size: 4096 (vs Qwen2-VL 3584); rm_head adapts automatically via config.hidden_size
"""

from typing import List, Optional

import torch
import torch.nn as nn

from transformers import Qwen3VLForConditionalGeneration


class Qwen3VLRewardModelBT(Qwen3VLForConditionalGeneration):
    """Bradley-Terry Reward Model on the Qwen3-VL backbone.

    forward outputs {"logits": [B, output_dim]}, used for pairwise ranking / STD training.
    """

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
        **kwargs,
    ):
        super().__init__(config)
        self.output_dim = output_dim

        # Obtain hidden_size (located in Qwen3-VL's text_config)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = config.text_config.hidden_size

        if rm_head_type == "default":
            self.rm_head = nn.Linear(hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                layers = []
                in_dim = hidden_size
                for layer_idx in range(rm_head_kwargs.get("num_layers", 3)):
                    if layer_idx < rm_head_kwargs.get("num_layers", 3) - 1:
                        out_dim = rm_head_kwargs["hidden_size"]
                        layers.append(nn.Linear(in_dim, out_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(rm_head_kwargs.get("dropout", 0.1)))
                        in_dim = out_dim
                    else:
                        layers.append(
                            nn.Linear(
                                in_dim,
                                output_dim,
                                bias=rm_head_kwargs.get("bias", False),
                            )
                        )
                self.rm_head = nn.Sequential(*layers)
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        self.rm_head.to(torch.float32)
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        # Used by the FiLMContinuous subclass; accepted but ignored here
        cond_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass: obtain backbone hidden states -> pooling -> rm_head.

        Qwen3-VL's self.model() handles vision encoding internally,
        so pixel_values and related arguments can be passed in directly.
        """
        # Qwen3-VL: self.model() accepts pixel_values and handles vision internally
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        hidden_states = outputs[0]  # [B, L, D]

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = self.rm_head(hidden_states)  # [B, L, output_dim]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # Sequence length for pooling
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id)
                    .int()
                    .argmax(-1)
                    - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        # Pooling
        if self.reward_token == "last":
            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(
                sequence_lengths, min=0, max=logits.size(1) - 1
            )
            pooled_logits = torch.stack(
                [
                    logits[i, : valid_lengths[i]].mean(dim=0)
                    for i in range(batch_size)
                ]
            )
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            pooled_hidden = hidden_states[special_token_mask, ...]
            pooled_hidden = pooled_hidden.view(batch_size, -1)
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                pooled_logits = self.rm_head(pooled_hidden)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

        return {"logits": pooled_logits}


class Qwen3VLRewardModelFiLMContinuous(Qwen3VLRewardModelBT):
    """Qwen3-VL + FiLM conditioning.

    Modulates pooled_hidden via cond_values [B, 2] (capability, iter_norm):
        conditioned = pooled_hidden * (1 + gamma) + beta
    where gamma and beta are produced by cond_encoder -> film_gen.

    Zero initialization ensures the initial behavior is identical to the base model.
    """

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
        cond_dim=256,
        **kwargs,
    ):
        super().__init__(
            config,
            output_dim=output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
            rm_head_type=rm_head_type,
            rm_head_kwargs=rm_head_kwargs,
            **kwargs,
        )

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = config.text_config.hidden_size

        # Condition encoder: [capability, iter_norm] -> cond_emb
        self.cond_encoder = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FiLM generator: cond_emb -> (gamma, beta)
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        # Zero initialization: initially gamma=0, beta=0 -> behavior identical to base model
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cond_values: Optional[torch.Tensor] = None,
        level_ids: Optional[torch.LongTensor] = None,
        iter_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Backbone forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        hidden_states = outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # Pooling -> pooled_hidden
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id)
                    .int()
                    .argmax(-1)
                    - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(
                sequence_lengths, min=0, max=hidden_states.size(1) - 1
            )
            pooled_hidden = torch.stack(
                [
                    hidden_states[i, : valid_lengths[i]].mean(dim=0)
                    for i in range(batch_size)
                ]
            )
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            pooled_hidden = hidden_states[special_token_mask, ...]
            pooled_hidden = pooled_hidden.view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

        # FiLM conditioning
        _LEVEL_TO_CAPABILITY = {0: 0.0, 1: 0.5, 2: 1.0}
        if cond_values is None:
            if level_ids is not None:
                caps = torch.tensor(
                    [_LEVEL_TO_CAPABILITY.get(l.item(), 0.5) for l in level_ids],
                    dtype=torch.float32,
                    device=pooled_hidden.device,
                )
                if iter_values is not None:
                    iters = iter_values.to(
                        device=pooled_hidden.device, dtype=torch.float32
                    )
                else:
                    iters = torch.zeros_like(caps)
                cond_values = torch.stack([caps, iters], dim=-1)
            elif iter_values is not None:
                caps = torch.full_like(
                    iter_values,
                    0.5,
                    dtype=torch.float32,
                    device=pooled_hidden.device,
                )
                iters = iter_values.to(
                    device=pooled_hidden.device, dtype=torch.float32
                )
                cond_values = torch.stack([caps, iters], dim=-1)
            else:
                cond_values = torch.tensor(
                    [[0.5, 0.0]] * batch_size,
                    dtype=torch.float32,
                    device=pooled_hidden.device,
                )

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            pooled_hidden_f32 = pooled_hidden.float()
            cond_values_f32 = cond_values.float()

            cond_emb = self.cond_encoder(cond_values_f32)
            film_params = self.film_gen(cond_emb)
            gamma, beta = film_params.chunk(2, dim=-1)

            conditioned = pooled_hidden_f32 * (1.0 + gamma) + beta
            logits = self.rm_head(conditioned)

        return {"logits": logits}


class CapabilityEncoder(nn.Module):
    """Infer capability (1D scalar) from the pooled features within a group."""

    def __init__(self, hidden_size: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, cond_dim)
        self.pool_attn = nn.Linear(cond_dim, 1)
        self.out_proj = nn.Linear(cond_dim, 1)

    def forward(self, group_features: torch.Tensor) -> torch.Tensor:
        """group_features: [k, hidden_size] -> scalar cap in [0, 1]."""
        proj_dtype = self.proj.weight.dtype
        x = self.proj(group_features.to(dtype=proj_dtype))   # [k, cond_dim]
        attn = torch.softmax(self.pool_attn(x), dim=0)       # [k, 1]
        group_emb = (x * attn).sum(dim=0)                    # [cond_dim]
        cap_pred = torch.sigmoid(self.out_proj(group_emb))    # [1]
        return cap_pred.squeeze(-1).to(dtype=torch.float32)   # scalar


class Qwen3VLRewardModelFiLMHybrid(Qwen3VLRewardModelFiLMContinuous):
    """Hybrid-condition FiLM: capability inferred implicitly from the image, rl_iter passed explicitly.

    Training procedure:
    1. Pre-train the CapabilityEncoder (frozen backbone) -> accurately predict capability
    2. Main training: FiLM uses [predicted_cap, explicit_iter]

    cond_values = [cap_encoder(pooled_hidden), iter_values]
    """

    def __init__(
        self,
        config,
        output_dim=2,
        reward_token="special",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
        cond_dim=256,
        **kwargs,
    ):
        super().__init__(
            config,
            output_dim=output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
            rm_head_type=rm_head_type,
            rm_head_kwargs=rm_head_kwargs,
            cond_dim=cond_dim,
            **kwargs,
        )
        # hidden_size: Qwen3VL uses text_config.hidden_size
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = config.text_config.hidden_size
        self.cap_encoder = CapabilityEncoder(hidden_size, cond_dim)
        self.cap_encoder.to(torch.float32)

    def _infer_capability(
        self, pooled_hidden, group_ids=None
    ):
        pooled_detached = pooled_hidden.detach()
        bsz = pooled_detached.shape[0]
        device = pooled_detached.device
        caps = torch.zeros(bsz, dtype=torch.float32, device=device)
        use_perimage = getattr(self, 'per_image_cap', False)
        if use_perimage or (group_ids is None and bsz == 1):
            for i in range(bsz):
                caps[i] = self.cap_encoder(pooled_detached[i:i+1])
        elif group_ids is not None:
            group_ids = group_ids.to(device).long()
            for gid in torch.unique(group_ids, sorted=True):
                mask = group_ids == gid
                caps[mask] = self.cap_encoder(pooled_detached[mask])
        else:
            pred = self.cap_encoder(pooled_detached)
            caps[:] = pred
        return caps

        group_ids = group_ids.to(device).long()
        unique_ids = torch.unique(group_ids, sorted=True)
        for gid in unique_ids:
            mask = group_ids == gid
            if not torch.any(mask):
                continue
            pred = self.cap_encoder(pooled_detached[mask])
            caps[mask] = pred
        return caps

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        cond_values: Optional[torch.Tensor] = None,
        iter_values: Optional[torch.Tensor] = None,
        level_ids: Optional[torch.LongTensor] = None,
        group_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # ---------- Backbone ----------
        # Qwen3-VL: self.model() handles vision encoding internally
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        hidden_states = outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        # ---------- Pooling ----------
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id)
                    .int()
                    .argmax(-1)
                    - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(
                sequence_lengths, min=0, max=hidden_states.size(1) - 1
            )
            pooled_hidden = torch.stack(
                [
                    hidden_states[i, : valid_lengths[i]].mean(dim=0)
                    for i in range(batch_size)
                ]
            )
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (
                    input_ids == special_token_id
                )
            pooled_hidden = hidden_states[special_token_mask, ...]
            pooled_hidden = pooled_hidden.view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

        # ---------- Hybrid condition: predicted cap + explicit iter ----------
        cap_pred = self._infer_capability(pooled_hidden, group_ids=group_ids)

        if iter_values is not None:
            iter_val = iter_values.to(device=pooled_hidden.device, dtype=torch.float32)
        else:
            iter_val = torch.zeros(batch_size, dtype=torch.float32, device=pooled_hidden.device)

        cond_values = torch.stack([cap_pred, iter_val], dim=-1)  # [batch, 2]

        # ---------- FiLM ----------
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            pooled_hidden_f32 = pooled_hidden.float()
            cond_values_f32 = cond_values.float()
            cond_emb = self.cond_encoder(cond_values_f32)
            film_params = self.film_gen(cond_emb)
            gamma, beta = film_params.chunk(2, dim=-1)
            conditioned = pooled_hidden_f32 * (1.0 + gamma) + beta
            logits = self.rm_head(conditioned)

        return {"logits": logits, "cond_pred": cond_values}
