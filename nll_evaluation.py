"""Score recorded next actions under teacher forcing using the training mask."""

from collections import defaultdict
from math import fsum, isfinite

import torch
from torch.nn import functional as F

from qwen_model import project_selected_tokens


def prepare_trajectory_nll(collator, trajectory):
    """Tokenize once and align each contiguous supervised span with an action."""
    batch = collator([trajectory])
    labels = batch["labels"][0].tolist()
    action_spans = []
    start = None
    for index, label in enumerate(labels):
        if label != -100 and start is None:
            start = index
        if label == collator.turn_end:
            action_spans.append((start, index + 1))
            start = None
        elif label == -100 and start is not None:
            raise ValueError("Supervised span ended before the assistant end token.")

    actions = [
        message["content"][0]["text"]
        for message in trajectory["messages"]
        if message["role"] == "assistant"
    ]
    step_ids = trajectory["step_ids"]
    if start is not None or not (len(action_spans) == len(actions) == len(step_ids)):
        raise ValueError(f"Action span count differs from step count for {trajectory['sample_id']}.")

    rows = []
    for step_id, action, (start, end) in zip(step_ids, actions, action_spans):
        if start == 0:
            raise ValueError("A supervised token needs a preceding position for causal prediction.")
        decoded = collator.processor.tokenizer.decode(
            labels[start:end - 1], clean_up_tokenization_spaces=False
        )
        if decoded != action:
            raise ValueError(f"Supervised text differs from the recorded action at step {step_id}.")
        rows.append({
            "sample_id": trajectory["sample_id"],
            "step_id": step_id,
            "target": action,
            "token_count": end - start,
        })
    return batch, rows


@torch.inference_mode()
def score_trajectory_nll(model, batch, action_rows):
    """Return each human action's NLL, including its assistant end token."""
    labels = batch["labels"][0].to(model.device)
    positions = (labels[1:] != -100).nonzero(as_tuple=True)[0]
    targets = labels[positions + 1]
    inputs = {key: value.to(model.device) for key, value in batch.items() if key != "labels"}

    # Position k-1 predicts token k. Keep the trajectory in causal context while
    # skipping the vocabulary projection for screenshots and other prompt tokens.
    with project_selected_tokens(model, positions), torch.autocast(
        model.device.type, dtype=model.dtype, enabled=model.device.type == "cuda"
    ):
        logits = model(**inputs, use_cache=False).logits[0]
    token_losses = F.cross_entropy(logits.float(), targets, reduction="none").tolist()
    if len(token_losses) != sum(row["token_count"] for row in action_rows):
        raise ValueError("Action token counts differ from the scored targets.")
    if not all(isfinite(loss) for loss in token_losses):
        raise ValueError("Non-finite action NLL.")

    rows = []
    offset = 0
    for row in action_rows:
        count = row["token_count"]
        end = offset + count
        total = fsum(token_losses[offset:end])
        rows.append({**row, "nll_sum": total, "nll_mean": total / count})
        offset = end
    return rows


def aggregate_nll(rows):
    """Summarize over tokens, actions, and equally weighted trajectories."""
    by_trajectory = defaultdict(list)
    for row in rows:
        by_trajectory[row["sample_id"]].append(row)
    total_tokens = sum(row["token_count"] for row in rows)
    total_nll = fsum(row["nll_sum"] for row in rows)
    # Average all target tokens within a trajectory before giving each
    # trajectory equal weight. The action mean below uses one vote per action.
    trajectory_means = [
        fsum(row["nll_sum"] for row in actions) / sum(row["token_count"] for row in actions)
        for actions in by_trajectory.values()
    ]
    return {
        "num_trajectories": len(trajectory_means),
        "num_actions": len(rows),
        "total_tokens": total_tokens,
        "nll_sum": total_nll,
        "token_mean_nll": total_nll / total_tokens,
        "mean_action_nll_sum": total_nll / len(rows),
        "mean_action_token_nll": fsum(row["nll_mean"] for row in rows) / len(rows),
        "mean_trajectory_token_nll": fsum(trajectory_means) / len(trajectory_means),
    }
