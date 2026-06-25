"""Helpers for explicit vs. implicit condition forwarding."""

from __future__ import annotations


def get_model_forward_strip_keys(model_type: str) -> set[str]:
    """Return batch keys that should not be forwarded into model(...).

    film_implicit: strip cond_values + iter_values (both inferred by GroupEncoder)
    film_hybrid:   strip cond_values + level_ids (cap inferred, iter explicit)
                   keep iter_values (explicit rl_iter) and group_ids (for CapabilityEncoder)
    """
    model_type = str(model_type or "base_rm").lower()
    if model_type == "film_implicit":
        return {"iter_values", "level_ids", "cond_values"}
    if model_type in ("film_hybrid", "dual_film", "iter_film_cap_residual"):
        # cap inferred by CapabilityEncoder, iter explicit
        # keep: iter_values, group_ids
        # strip: level_ids, cond_values
        return {"level_ids", "cond_values"}
    return {"iter_values", "group_ids", "level_ids"}
