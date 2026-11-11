"""Load prepared examples and express them as multimodal SFT conversations."""

import argparse
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).parent / "data" / "processed"


def browser_turn(prompt, action):
    return [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": action}]},
    ]


def load_screenshot(directory, state):
    with Image.open(directory / state["screenshot"]) as source:
        return source.convert("RGB")


def make_messages(example):
    state = example["current_state"]
    history = "\n".join(
        f"{index}. {action}"
        for index, action in enumerate(example["previous_actions"], start=1)
    ) or "(none)"
    prompt = (
        f"Task:\n{example['task']}\n\n"
        f"Screenshot size: {state['width']} x {state['height']} pixels\n\n"
        f"Previous actions:\n{history}\n\n"
        "Predict the next browser action. Return only its action string. "
        "Use coordinates in the original screenshot's pixel space."
    )
    return browser_turn(prompt, example["target_action"])


class BrowserActionDataset(Dataset):
    """Per-action format used for the training timing comparison."""

    def __init__(self, jsonl_path):
        self.path = Path(jsonl_path)
        with self.path.open(encoding="utf-8") as stream:
            self.examples = [json.loads(line) for line in stream]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return {
            "sample_id": example["sample_id"],
            "step_id": example["step_id"],
            "messages": make_messages(example),
            "images": [load_screenshot(self.path.parent, example["current_state"])],
        }


class BrowserTrajectoryDataset(BrowserActionDataset):
    """One conversation per trajectory, with one screenshot/action pair per turn."""

    def __init__(self, jsonl_path):
        super().__init__(jsonl_path)
        grouped = {}
        for example in self.examples:
            grouped.setdefault(example["sample_id"], []).append(example)
        self.trajectories = [
            sorted(steps, key=lambda step: step["step_id"])
            for steps in grouped.values()
        ]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        steps = self.trajectories[index]
        first = steps[0]
        initial_actions = "\n".join(first["previous_actions"]) or "(none)"
        introduction = (
            f"Task:\n{first['task']}\n\n"
            f"Initial actions:\n{initial_actions}\n\n"
            "Predict the next browser action. Return only its action string. "
            "Use coordinates in the original screenshot's pixel space.\n\n"
        )
        messages = []
        images = []
        for position, step in enumerate(steps):
            state = step["current_state"]
            text = f"Screenshot size: {state['width']} x {state['height']} pixels"
            # Later turns inherit the task and action history from the conversation.
            if position == 0:
                text = introduction + text
            messages.extend(browser_turn(text, step["target_action"]))
            images.append(load_screenshot(self.path.parent, state))
        return {
            "sample_id": first["sample_id"],
            "step_ids": [step["step_id"] for step in steps],
            "num_actions": len(steps),
            "messages": messages,
            "images": images,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--unit", choices=["action", "trajectory"], default="trajectory")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    dataset_class = BrowserTrajectoryDataset if args.unit == "trajectory" else BrowserActionDataset
    dataset = dataset_class(args.data_dir / f"{args.split}.jsonl")
    example = dataset[args.index]
    print(f"{len(dataset)} {args.unit} examples; {len(example['images'])} images")
    print(json.dumps(example["messages"], ensure_ascii=False, indent=2))
