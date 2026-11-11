"""Compare base and LoRA checkpoints on the same held-out action targets."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

from peft import PeftModel
import torch

from cached_generation import generate_cached_tokens, prepare_cached_generation, predict_cached_samples
from collator import ActionCollator
from config import load_config, save_config
from data import BrowserTrajectoryDataset
from evaluation import DEFAULT_SETTINGS, score_action
from infer import encode_prefix, generate_tokens, load_model, make_prefix, predict_actions
from nll_evaluation import aggregate_nll, prepare_trajectory_nll, score_trajectory_nll
from qwen_model import project_selected_tokens
from sampled_metrics import aggregate_sampled_scores


CACHE_TV_TOLERANCE = 0.05


def load_comparison_model(config):
    base_config = {**config, "inference": {**config["inference"], "adapter_path": None}}
    base, processor = load_model(base_config)
    checkpoints = iter(config["evaluation"]["checkpoints"].items())
    name, path = next(checkpoints)
    model = PeftModel.from_pretrained(base, path, adapter_name=name, is_trainable=False)
    for name, path in checkpoints:
        model.load_adapter(path, adapter_name=name, is_trainable=False)
    return model.eval().requires_grad_(False), processor


def adapter_context(model, name):
    if name == "base":
        return model.disable_adapter()
    model.set_adapter(name, inference_mode=True)
    return nullcontext()


def run_nll(model, processor, dataset, modes, settings, output_dir):
    collator = ActionCollator(processor)
    results = {mode: [] for mode in modes}
    started = time.perf_counter()
    with (output_dir / "action_nll.jsonl").open("w", encoding="utf-8") as stream:
        for index, trajectory in enumerate(dataset, start=1):
            batch, spans = prepare_trajectory_nll(collator, trajectory)
            trajectory_means = []
            for mode in modes:
                with adapter_context(model, mode):
                    rows = score_trajectory_nll(model, batch, spans)
                results[mode].extend(rows)
                for row in rows:
                    stream.write(json.dumps({"model": mode, **row}, ensure_ascii=False) + "\n")
                trajectory_means.append(f"{mode}={aggregate_nll(rows)['mean_action_token_nll']:.4f}")
            stream.flush()
            print(f"NLL {index}/{len(dataset)} trajectories, {len(spans)} actions, "
                  + ", ".join(trajectory_means), flush=True)
    summary = {
        "split": settings["split"],
        "primary_metric": "mean_action_token_nll",
        "target": "Recorded next action with its assistant end token",
        "checkpoints": settings["checkpoints"],
        "elapsed_seconds": time.perf_counter() - started,
        "models": {mode: aggregate_nll(rows) for mode, rows in results.items()},
    }
    (output_dir / "nll_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def shared_decode_distance(cached, direct, temperature):
    # After the first differing token, the continuations condition on different histories.
    cached_ids = cached.sequences[0, -len(cached.scores):].tolist()
    direct_ids = direct.sequences[0, -len(direct.scores):].tolist()
    steps = 0
    for cached_id, direct_id in zip(cached_ids, direct_ids):
        steps += 1
        if cached_id != direct_id:
            break
    scores = torch.stack((torch.cat(cached.scores[:steps]), torch.cat(direct.scores[:steps])), dim=1)
    probabilities = (scores.float() / temperature).softmax(dim=-1)
    max_tv = (probabilities[:, 0] - probabilities[:, 1]).abs().sum(dim=-1).max().item() / 2
    return steps, max_tv


def check_cached_generation(model, processor, dataset, modes, inference):
    """Compare the conditional distributions along a shared greedy prefix."""
    collator = ActionCollator(processor)
    check_index = min(range(len(dataset)), key=lambda index: abs(len(dataset.trajectories[index]) - 15))
    trajectory = dataset[check_index]
    batch = collator([trajectory])
    positions = [0, len(trajectory["step_ids"]) // 2, len(trajectory["step_ids"]) - 1]
    greedy = {**inference, "do_sample": False, "max_new_tokens": min(32, inference["max_new_tokens"])}
    temperature = inference.get("temperature", 1.0) if inference["do_sample"] else 1.0
    max_distance = 0.0
    checked_tokens = 0
    for mode in modes:
        with adapter_context(model, mode):
            prepared = prepare_cached_generation(model, batch)
            for position in positions:
                cached = generate_cached_tokens(model, prepared, position, greedy, 1, with_scores=True)
                prefix = encode_prefix(model, processor, **make_prefix(trajectory, position))
                with project_selected_tokens(model, slice(-1, None)):
                    direct = generate_tokens(model, prefix, greedy, with_scores=True)
                del prefix
                steps, distance = shared_decode_distance(cached, direct, temperature)
                max_distance = max(max_distance, distance)
                checked_tokens += steps
                print(f"Cache check {mode}, step {trajectory['step_ids'][position]}: "
                      f"{steps} tokens, max TV={distance:.4f}", flush=True)
            del prepared
    print(f"Cache check across {checked_tokens} tokens: max TV={max_distance:.4f}", flush=True)
    return {"compared_tokens": checked_tokens, "max_total_variation": max_distance}


def sample_trajectory(model, processor, trajectory, batch, settings, num_samples, seed):
    prepared = prepare_cached_generation(model, batch) if batch is not None else None
    for position in range(len(trajectory["step_ids"])):
        # The same recorded prefix starts from the same seed for every model.
        torch.manual_seed(seed + position)
        if prepared is not None:
            yield predict_cached_samples(model, processor, prepared, position, settings, num_samples)
        else:
            yield predict_actions(
                model, processor, **make_prefix(trajectory, position), settings=settings,
                num_samples=num_samples,
            )


def run_generation(model, processor, dataset, modes, config, output_dir):
    inference, settings = config["inference"], config["evaluation"]
    thresholds = {key: settings.get(key, default) for key, default in DEFAULT_SETTINGS.items()}
    num_samples = settings.get("samples_per_prefix", 1)
    seed = config["training"]["seed"]
    if num_samples < 1 or (num_samples > 1 and not inference["do_sample"]):
        raise ValueError("Multiple samples per prefix require stochastic decoding.")
    started = time.perf_counter()
    print(f"Checking generation on {len(dataset)} {settings['split']} trajectories; "
          f"{num_samples} draw(s) per prefix and model.", flush=True)
    cache_check = settings.get("reuse_causal_cache", True) and check_cached_generation(
        model, processor, dataset, modes, inference
    )
    use_cache = bool(cache_check and cache_check["max_total_variation"] <= CACHE_TV_TOLERANCE)
    if not use_cache:
        print("Using independent prefixes.", flush=True)
    collator = ActionCollator(processor)
    rows = []
    prefixes_done = 0
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for index, trajectory in enumerate(dataset):
            steps = dataset.trajectories[index]
            batch = collator([trajectory]) if use_cache else None
            for mode in modes:
                with adapter_context(model, mode):
                    samples = sample_trajectory(
                        model, processor, trajectory, batch, inference, num_samples, seed + prefixes_done
                    )
                    for record, predictions in zip(steps, samples, strict=True):
                        state = record["current_state"]
                        target = record["target_action"]
                        for sample_index, prediction in enumerate(predictions):
                            result = score_action(prediction, target, state["width"], state["height"], thresholds)
                            row = {
                                "model": mode, "sample_id": record["sample_id"], "step_id": record["step_id"],
                                "sample_index": sample_index, "target": target,
                                "prediction": prediction, "scores": result,
                            }
                            rows.append(row)
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                print(f"Generation {index + 1}/{len(dataset)} trajectories, "
                      f"{mode}: {prefixes_done + len(steps)} prefixes", flush=True)
            prefixes_done += len(steps)
    summary = {
        "protocol": "Next action from the recorded human prefix and current screenshot",
        "split": settings["split"],
        "num_trajectories": len(dataset),
        "num_prefixes": prefixes_done,
        "cached_generation": use_cache,
        "cache_check": cache_check,
        "checkpoints": settings["checkpoints"],
        "sampling": {
            "samples_per_prefix": num_samples, "seed": seed,
            **{key: inference[key] for key in ("do_sample", "temperature", "top_p", "max_new_tokens")
               if key in inference},
        },
        "thresholds": thresholds,
        "elapsed_seconds": time.perf_counter() - started,
        **aggregate_sampled_scores(rows),
    }
    (output_dir / "generation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for mode, result in summary["models"].items():
        print(f"{mode}: {result['count']} predictions, "
              f"type_accuracy={result['type_accuracy']:.4f}, joint_match={result['joint_match_rate']:.4f}")


def run(config):
    settings = config["evaluation"]
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = settings.get("phase", "nll")
    if phase not in {"nll", "generation"}:
        raise ValueError(f"Unknown comparison phase: {phase}")
    save_config(config, output_dir / f"{phase}_config.yaml")
    dataset = BrowserTrajectoryDataset(Path(config["data"]["directory"]) / f"{settings['split']}.jsonl")
    model, processor = load_comparison_model(config)
    modes = ["base", *settings["checkpoints"]]
    if phase == "nll":
        run_nll(model, processor, dataset, modes, settings, output_dir)
    else:
        run_generation(model, processor, dataset, modes, config, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config))
