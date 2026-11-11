"""Summarize repeated next-action predictions by sample and recorded prefix."""

from collections import Counter, defaultdict
from math import isfinite
from statistics import fmean

from evaluation import aggregate_scores


_STATUS_FIELDS = {"parse_valid", "type_match", "parameters_valid", "complete_match", "joint_match"}


def _add_scroll_squared_error(score):
    error = score.get("scroll_magnitude_error_normalized")
    if error is not None:
        squared_error = error * error
        if isfinite(squared_error):
            return {**score, "scroll_magnitude_squared_error_normalized": squared_error}
    return score


def _prefix_summary(prefixes):
    valid_draws = [
        sum(score["type_match"] and score["parameters_valid"] for score in group)
        for group in prefixes
    ]
    return {
        "prefix_count": len(prefixes),
        # Any-of-K counts each recorded prefix once, regardless of sample count.
        "any_type_match_rate": fmean(any(score["type_match"] for score in group) for group in prefixes),
        "any_joint_match_rate": fmean(any(score["joint_match"] for score in group) for group in prefixes),
        "valid_same_type_count": sum(valid_draws),
        "valid_same_type_prefix_count": sum(count > 0 for count in valid_draws),
    }


def _shared_parameter_summary(grouped, target_types):
    """Give each commonly scorable prefix one vote for a parameter metric."""
    # Index by metric, then prefix, so common support is visible at each prefix.
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model, prefixes in grouped.items():
        for prefix, scores in prefixes.items():
            for score in scores:
                if not (score["type_match"] and score["parameters_valid"]):
                    continue
                for metric, value in score.items():
                    if metric not in _STATUS_FIELDS and isinstance(value, (int, float)) and isfinite(value):
                        values[score["target_type"], metric][prefix][model].append(value)

    summary = {name: {} for name in sorted(set(target_types.values()))}
    for (action_type, metric), by_prefix in sorted(values.items()):
        shared = [by_model for by_model in by_prefix.values() if len(by_model) == len(grouped)]
        means = {}
        for model in grouped:
            prefix_samples = [by_model[model] for by_model in shared]
            means[model] = {
                "mean": fmean(fmean(samples) for samples in prefix_samples) if shared else None,
                "sample_count": sum(map(len, prefix_samples)),
            }
        summary[action_type][metric] = {"prefix_count": len(shared), "models": means}
    return summary


def aggregate_sampled_scores(rows):
    """Aggregate score_action rows with model, sample_id, step_id, and sample_index."""
    grouped = defaultdict(lambda: defaultdict(list))
    target_types = {}
    seen = set()
    for row in rows:
        model = row["model"]
        prefix = row["sample_id"], row["step_id"]
        key = model, prefix, row["sample_index"]
        if key in seen:
            raise ValueError(f"Duplicate generated sample: {key}")
        seen.add(key)
        score = _add_scroll_squared_error(row["scores"])
        if target_types.setdefault(prefix, score["target_type"]) != score["target_type"]:
            raise ValueError(f"Reference action type differs for prefix: {prefix}")
        grouped[model][prefix].append(score)

    models = {}
    sample_counts = {}
    for model, by_prefix in grouped.items():
        model_scores = aggregate_scores(score for scores in by_prefix.values() for score in scores)
        model_scores.update(_prefix_summary(by_prefix.values()))
        for action_type, metrics in model_scores["per_type"].items():
            metrics.update(_prefix_summary([
                scores for prefix, scores in by_prefix.items()
                if target_types[prefix] == action_type
            ]))
        models[model] = model_scores
        sample_counts[model] = dict(sorted(Counter(map(len, by_prefix.values())).items()))

    return {
        "models": models,
        "shared_prefix_parameters": _shared_parameter_summary(grouped, target_types),
        "sample_counts_per_prefix": sample_counts,
    }
