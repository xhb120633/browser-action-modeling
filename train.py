"""Local Qwen2.5-VL supervised fine-tuning with TRL and language-layer LoRA."""

import argparse
import json
from pathlib import Path

import peft
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import ConcatDataset, Subset
import transformers
from transformers import set_seed
import trl
from trl import SFTConfig

from collator import ActionCollator
from config import DEFAULT_CONFIG, load_config, save_config
from data import BrowserActionDataset, BrowserTrajectoryDataset
from qwen_model import encode_images_separately, load_base_model, load_processor
from trajectory_trainer import TrajectoryTrainer


def save_input_example(output_dir, example, collator, config):
    """Show the actual action tokens selected by the training mask."""
    batch = collator([example])
    labels = batch["labels"][0]
    action_tokens = labels[labels != -100]
    example_report = {
        "sample_id": example["sample_id"],
        "unit": config["data"]["unit"],
        "microbatch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "image_token_budget": config["model"]["max_image_tokens"],
        "step_ids": example.get("step_ids", [example.get("step_id")]),
        "messages": example["messages"],
        "image_sizes": [image.size for image in example["images"]],
        "image_grid_thw": batch["image_grid_thw"].tolist(),
        "input_tokens": int(batch["attention_mask"].sum()),
        "supervised_tokens": len(action_tokens),
        "supervised_text": collator.processor.tokenizer.decode(action_tokens),
    }
    (output_dir / "input_example.json").write_text(
        json.dumps(example_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"First {example_report['unit']}: {len(example['images'])} screenshots, "
          f"{example_report['input_tokens']} input tokens, "
          f"{example_report['supervised_tokens']} action tokens", flush=True)


def main(config, config_path=None):
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_resolved.yaml"
    save_config(config, snapshot_path)
    set_seed(training_config["seed"])
    print(f"Configuration saved to {snapshot_path}", flush=True)

    processor = load_processor(model_config)
    collator = ActionCollator(processor)
    unit = data_config["unit"]
    dataset_class = BrowserTrajectoryDataset if unit == "trajectory" else BrowserActionDataset
    data_dir = Path(data_config["directory"])
    train_splits = data_config.get("train_splits", ["train"])
    validation_split = data_config.get("validation_split", "val")
    if validation_split in train_splits:
        raise ValueError("The validation split must be separate from the training splits.")
    train_data = ConcatDataset([dataset_class(data_dir / f"{split}.jsonl") for split in train_splits])
    val_data = dataset_class(data_dir / f"{validation_split}.jsonl") if validation_split else None
    if data_config["train_samples"]:
        train_data = Subset(train_data, range(min(data_config["train_samples"], len(train_data))))
    if val_data is not None and data_config["val_samples"]:
        val_data = Subset(val_data, range(min(data_config["val_samples"], len(val_data))))

    save_input_example(output_dir, train_data[0], collator, config)

    model = load_base_model(model_config, use_cache=False)
    model = get_peft_model(model, LoraConfig(**config["lora"]))
    model.print_trainable_parameters()
    encode_images_separately(model.visual)

    trainer = TrajectoryTrainer(
        model=model,
        args=SFTConfig(**training_config),
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    before = trainer.evaluate() if val_data is not None else None
    result = trainer.train()
    after = trainer.evaluate() if val_data is not None else None

    trainer.save_model(str(output_dir))
    processor.save_pretrained(output_dir)
    trainer.save_state()
    report = {
        "model": model_config["path"],
        "config_source": str(Path(config_path).resolve()) if config_path else None,
        "config_snapshot": str(snapshot_path),
        "unit": unit,
        "train_splits": train_splits,
        "validation_split": validation_split,
        "microbatch_size": training_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "image_token_budget": model_config["max_image_tokens"],
        "versions": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "trl": trl.__version__, "peft": peft.__version__,
        },
        "train_examples": len(train_data),
        "validation_examples": len(val_data) if val_data is not None else 0,
        "optimizer_steps": trainer.state.global_step,
        "train_metrics": result.metrics,
        "validation_before": before,
        "validation_after": after,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    main(load_config(args.config), config_path=args.config)
