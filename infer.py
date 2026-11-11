"""Generate next actions from recorded prefixes and score reference agreement."""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import time

from peft import PeftModel
import torch
from transformers import set_seed

from config import DEFAULT_CONFIG, load_config, save_config
from data import BrowserTrajectoryDataset
from evaluation import DEFAULT_SETTINGS, aggregate_scores, score_action
from qwen_model import encode_images_separately, load_base_model, load_processor, project_selected_tokens


def make_prefix(trajectory, position):
    """End at the current user screenshot; withhold its action and all later data."""
    return {
        "messages": trajectory["messages"][:2 * position + 1],
        "images": trajectory["images"][:position + 1],
    }


def select_examples(dataset, limit):
    """Limited runs cover available action classes, then rotate trajectories."""
    by_type = defaultdict(list)
    all_examples = []
    for trajectory_index, steps in enumerate(dataset.trajectories):
        for position, step in enumerate(steps):
            reference = (trajectory_index, position)
            by_type[step["action_name"]].append(reference)
            all_examples.append(reference)
    if limit == 0 or limit >= len(all_examples):
        return all_examples, "full_split"

    selected = []
    used_trajectories = set()
    for references in by_type.values():
        reference = next(
            (item for item in references if item[0] not in used_trajectories), references[0]
        )
        selected.append(reference)
        used_trajectories.add(reference[0])
    selected_set = set(selected)
    for position in range(max(map(len, dataset.trajectories))):
        for trajectory_index, steps in enumerate(dataset.trajectories):
            reference = (trajectory_index, position)
            if position < len(steps) and reference not in selected_set:
                selected.append(reference)
                selected_set.add(reference)
    return selected[:limit], "class_balanced_smoke"


def load_model(config):
    settings = config["model"]
    processor = load_processor(settings)
    model = load_base_model(settings, use_cache=True)
    adapter_path = config["inference"].get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.cuda().eval()
    model.requires_grad_(False)
    encode_images_separately(model.visual)
    return model, processor


def encode_prefix(model, processor, messages, images):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=images, return_tensors="pt")
    return batch.to(model.device)


def generate_tokens(model, inputs, settings, *, with_scores=False):
    options = {
        "max_new_tokens": settings["max_new_tokens"],
        "do_sample": settings["do_sample"],
        "use_cache": True,
    }
    if settings["do_sample"]:
        options.update({key: settings[key] for key in ("temperature", "top_p") if key in settings})
    if with_scores:
        options.update(return_dict_in_generate=True, output_scores=True)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=model.dtype, enabled=model.device.type == "cuda"
    ):
        return model.generate(**inputs, **options)


def predict_actions(model, processor, messages, images, settings, num_samples):
    batch = encode_prefix(model, processor, messages, images)
    prompt_length = batch["input_ids"].shape[1]
    # Independent prefixes serve as the simple inference path and cache fallback.
    with project_selected_tokens(model, slice(-1, None)):
        outputs = [
            generate_tokens(model, batch, settings)[0, prompt_length:] for _ in range(num_samples)
        ]
    return [processor.tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


def predict_action(model, processor, messages, images, settings):
    return predict_actions(model, processor, messages, images, settings, num_samples=1)[0]


def run(config, config_path=None):
    settings = config["evaluation"]
    inference = config["inference"]
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config_resolved.yaml")
    dataset_path = Path(config["data"]["directory"]) / f"{settings['split']}.jsonl"
    dataset = BrowserTrajectoryDataset(dataset_path)
    references, selection = select_examples(dataset, settings["max_examples"])
    trajectories = {index: dataset[index] for index in sorted({index for index, _ in references})}
    model, processor = load_model(config)
    adapter_loaded = bool(inference.get("adapter_path"))
    modes = ["adapter"] if adapter_loaded else ["base"]
    if adapter_loaded and inference["compare_base"]:
        modes.insert(0, "base")
    thresholds = {key: settings.get(key, default) for key, default in DEFAULT_SETTINGS.items()}
    scores = {mode: [] for mode in modes}

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for example_index, (trajectory_index, position) in enumerate(references):
            trajectory = trajectories[trajectory_index]
            record = dataset.trajectories[trajectory_index][position]
            prefix = make_prefix(trajectory, position)
            state = record["current_state"]
            for mode in modes:
                # Both models receive exactly the same recorded human history.
                context = model.disable_adapter() if mode == "base" and adapter_loaded else nullcontext()
                set_seed(config["training"]["seed"] + example_index)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with context:
                    prediction = predict_action(model, processor, **prefix, settings=inference)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                result = score_action(
                    prediction, record["target_action"], state["width"], state["height"], thresholds
                )
                scores[mode].append(result)
                output = {
                    "model": mode,
                    "sample_id": record["sample_id"],
                    "step_id": record["step_id"],
                    "screenshots_in_prefix": len(prefix["images"]),
                    "screenshot_width": state["width"],
                    "screenshot_height": state["height"],
                    "target": record["target_action"],
                    "prediction": prediction,
                    "generation_seconds": seconds,
                    "scores": result,
                }
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{example_index + 1}/{len(references)} {mode} "
                      f"({seconds:.2f}s): {prediction}", flush=True)

    summary = {
        "protocol": "One-step generation from recorded human prefixes.",
        "split": settings["split"],
        "selection": selection,
        "selection_note": (
            "All recorded steps in the split."
            if selection == "full_split"
            else "Coverage-biased smoke test, not a natural-distribution estimate."
        ),
        "examples": len(references),
        "trajectories": len(trajectories),
        "base_model": config["model"]["path"],
        "adapter_path": inference.get("adapter_path"),
        "image_token_budget": config["model"]["max_image_tokens"],
        "config_source": str(config_path) if config_path else None,
        "thresholds": thresholds,
        "generation": {
            "max_new_tokens": inference["max_new_tokens"],
            "do_sample": inference["do_sample"],
        },
        "models": {mode: aggregate_scores(values) for mode, values in scores.items()},
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(load_config(args.config), config_path=args.config)
