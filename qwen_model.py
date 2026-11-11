"""Shared Qwen model setup and memory-saving operations."""

from contextlib import contextmanager

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_processor(settings):
    pixels_per_visual_token = 28 * 28
    return AutoProcessor.from_pretrained(
        settings["path"],
        min_pixels=settings["min_image_tokens"] * pixels_per_visual_token,
        max_pixels=settings["max_image_tokens"] * pixels_per_visual_token,
        use_fast=settings["use_fast_processor"],
    )


def load_base_model(settings, *, use_cache):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings["path"],
        torch_dtype=getattr(torch, settings["torch_dtype"]),
        attn_implementation=settings["attn_implementation"],
    )
    # Transformers 4.51 reads the stored window size even when disabled.
    # Clear it to keep full causal attention on the SDPA fast path.
    if not model.config.use_sliding_window:
        model.config.sliding_window = None
    model.config.use_cache = use_cache
    return model


def encode_images_separately(visual):
    """Run the frozen encoder per image to avoid a mask over all patches."""
    original_forward = visual.forward

    def forward(hidden_states, grid_thw):
        patch_counts = grid_thw.prod(dim=1).tolist()
        images = torch.split(hidden_states, patch_counts)
        features = [original_forward(image, grid) for image, grid in zip(images, grid_thw.split(1))]
        return torch.cat(features, dim=0)

    visual.forward = forward


@contextmanager
def project_selected_tokens(model, positions):
    """Project selected hidden states to vocabulary logits after full attention."""
    head = model.get_output_embeddings()
    handle = head.register_forward_pre_hook(lambda _module, args: (args[0][:, positions, :],))
    try:
        yield
    finally:
        handle.remove()
