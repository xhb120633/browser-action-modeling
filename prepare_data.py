"""Split whole trajectories, then expand them into next-action examples."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


DATA_DIR = Path(__file__).parent / "data"


def trajectory_examples(row, output_dir: Path):
    """Use the current screenshot and only actions strictly before this step."""
    task = json.loads(row["instruction"])["low_level"]
    steps = json.loads(row["trajectory"])
    image_bytes = dict(zip(row["image_paths"], row["images"]))
    history = []

    for index, step_id in enumerate(sorted(steps, key=int)):
        step = steps[step_id]
        action = step["action"]
        action_name = action["action_output"]["action_name"]
        target = action["action_str"]

        # Initial navigation is setup, but still belongs in later action history.
        if index == 0 and action_name == "goto":
            history.append(target)
            continue

        image_path = Path("images") / row["sample_id"] / Path(step["screenshot"]).name
        destination = output_dir / image_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes[step["screenshot"]])

        yield {
            "sample_id": row["sample_id"],
            "step_id": int(step_id),
            "task": task,
            "current_state": {
                "screenshot": image_path.as_posix(),
                "width": step["image_w"],
                "height": step["image_h"],
            },
            "previous_actions": list(history),
            "target_action": target,
            "action_name": action_name,
        }
        history.append(target)


def prepare_data(raw_dir: Path, output_dir: Path, val_fraction=0.1, test_fraction=0.1, seed=42):
    files = sorted(raw_dir.glob("part-*.parquet"))
    ids = [
        sample_id
        for file in files
        for sample_id in pq.read_table(file, columns=["sample_id"])["sample_id"].to_pylist()
    ]
    assert len(ids) >= 3 and len(ids) == len(set(ids)), "Need unique trajectory IDs."
    random.Random(seed).shuffle(ids)
    val_count = max(1, round(len(ids) * val_fraction))
    test_count = max(1, round(len(ids) * test_fraction))
    assert val_count + test_count < len(ids), "Keep some trajectories for training."
    split_ids = {
        "train": ids[val_count + test_count:],
        "val": ids[:val_count],
        "test": ids[val_count:val_count + test_count],
    }
    split_by_id = {sample_id: split for split, values in split_ids.items() for sample_id in values}
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    actions = {split: Counter() for split in split_ids}

    with (
        (output_dir / "train.jsonl").open("w", encoding="utf-8") as train_file,
        (output_dir / "val.jsonl").open("w", encoding="utf-8") as val_file,
        (output_dir / "test.jsonl").open("w", encoding="utf-8") as test_file,
    ):
        handles = {"train": train_file, "val": val_file, "test": test_file}
        for file in files:
            for row in pq.read_table(file).to_pylist():
                split = split_by_id[row["sample_id"]]
                for example in trajectory_examples(row, output_dir):
                    handles[split].write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts[split] += 1
                    actions[split][example["action_name"]] += 1
            print(f"Processed {file.name}: {dict(counts)} examples", flush=True)

    summary = {
        "seed": seed,
        "instruction_variant": "low_level",
        "initial_goto": "excluded as target; included in subsequent history",
        "trajectories": {split: len(values) for split, values in split_ids.items()},
        "examples": dict(counts),
        "action_counts": {split: dict(values) for split, values in actions.items()},
        "quality_note": "Format processed; task success has not been manually audited.",
    }
    (output_dir / "splits.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "processed")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_data(args.raw_dir, args.output_dir, args.val_fraction, args.test_fraction, args.seed)
