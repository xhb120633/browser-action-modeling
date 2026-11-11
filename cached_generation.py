"""Reuse a Qwen2.5-VL trajectory's causal keys and values for prefix generation."""

from dataclasses import dataclass

from peft import PeftModel
import torch
from transformers.cache_utils import DynamicCache

from infer import generate_tokens
from qwen_model import project_selected_tokens


@dataclass
class CachedTrajectory:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_starts: list[int]
    rope_deltas: torch.Tensor
    past_key_values: tuple


def prepare_cached_generation(model, batch):
    """Prefill one collated trajectory under the adapter used for its decoding."""
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("Cached generation expects one trajectory per batch.")
    supervised = batch["labels"][0] != -100
    previous = torch.cat([supervised.new_zeros(1), supervised[:-1]])
    action_starts = (supervised & ~previous).nonzero(as_tuple=True)[0].tolist()
    inputs = {name: tensor.to(model.device) for name, tensor in batch.items() if name != "labels"}
    qwen = model.get_base_model() if isinstance(model, PeftModel) else model
    # Positions of earlier tokens are invariant to appended turns. Each action
    # prefix ends in the text of the assistant header, with equal T/H/W IDs.
    position_ids, _ = qwen.get_rope_index(
        inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )
    ends = torch.tensor(action_starts, device=model.device)
    rope_deltas = position_ids[0, :, ends - 1] + 1 - ends
    with project_selected_tokens(model, slice(-1, None)), torch.inference_mode(), torch.autocast(
        "cuda", dtype=model.dtype, enabled=model.device.type == "cuda"
    ):
        output = model(**inputs, position_ids=position_ids, use_cache=True)
    return CachedTrajectory(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        action_starts=action_starts, rope_deltas=rope_deltas,
        past_key_values=output.past_key_values.to_legacy_cache(),
    )


def predict_cached_samples(model, processor, prepared, position, settings, num_samples):
    """Branch at a recorded prefix without exposing its target or later turns."""
    prompt_length = prepared.action_starts[position]
    output = generate_cached_tokens(model, prepared, position, settings, num_samples)
    return [
        processor.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True).strip()
        for sequence in output
    ]


def generate_cached_tokens(model, prepared, position, settings, num_samples, *, with_scores=False):
    prompt_length = prepared.action_starts[position]
    # Keep the final header token uncached so generate computes its next-token
    # logits. Appending in DynamicCache uses cat, leaving the full cache intact.
    cached_length = prompt_length - 1
    cache = DynamicCache.from_legacy_cache(tuple(
        (key[..., :cached_length, :], value[..., :cached_length, :])
        for key, value in prepared.past_key_values
    ))
    # GenerationMixin expands tensor inputs but leaves DynamicCache untouched.
    if num_samples > 1:
        cache.batch_repeat_interleave(num_samples)
    inputs = {
        "input_ids": prepared.input_ids[:, :prompt_length].expand(num_samples, -1),
        "attention_mask": prepared.attention_mask[:, :prompt_length].expand(num_samples, -1),
        "past_key_values": cache,
    }
    qwen = model.get_base_model() if isinstance(model, PeftModel) else model
    saved_delta = qwen.rope_deltas
    qwen.rope_deltas = prepared.rope_deltas[:, position:position + 1].expand(num_samples, -1)
    try:
        return generate_tokens(model, inputs, settings, with_scores=with_scores)
    finally:
        qwen.rope_deltas = saved_delta
