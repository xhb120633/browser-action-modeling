# Predicting the next human browser action

A multimodal supervised fine-tuning pipeline for MolmoWeb-HumanTrajs. The
[technical note](NOTE.md) explains the input, loss, inference, and validation results.

## Run locally

The experiments ran on an RTX 4090 with Python 3.11 and PyTorch 2.6.0
with CUDA 12.6. Install the dependencies in an activated Python 3.11 environment.
[PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v260)

~~~sh
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python download_data.py --manifest configs/data-manifest.json
python prepare_data.py
python download_model.py --revision 66285546d2b821cf421d4f5eb2576359d3770cd3
python train.py --config config.yaml
python infer.py --config config.yaml
~~~

The default YAML trains for 20 updates and scores a small validation sample.
Model, data, training, and evaluation settings live in `config.yaml`. Training
writes the adapter, processor, resolved config, a training input example, and a
run report. Set `inference.adapter_path` to null for base-only inference.

## Reading the code

Start with [config.yaml](config.yaml), then follow one example through the pipeline.

| Stage | Files | What to look for |
| --- | --- | --- |
| Data preparation | [download_data.py](download_data.py), [prepare_data.py](prepare_data.py) | Download trajectories, join screenshots to actions, and split by trajectory. |
| Model input | [data.py](data.py), [collator.py](collator.py) | Turn each trajectory into screenshot/action messages and mask the loss to action tokens. |
| Training | [train.py](train.py), [trajectory_trainer.py](trajectory_trainer.py) | Set up LoRA, run training, save the adapter, and compute causal action loss. |
| Shared model | [qwen_model.py](qwen_model.py) | Load Qwen and its processor, and project only the needed hidden states to vocabulary logits. |
| Inference | [infer.py](infer.py), [cached_generation.py](cached_generation.py) | Generate a next action from a recorded prefix. The cache helper reuses overlapping prefixes for the full comparison. |
| Evaluation | [compare_checkpoints.py](compare_checkpoints.py), [nll_evaluation.py](nll_evaluation.py), [evaluation.py](evaluation.py), [sampled_metrics.py](sampled_metrics.py) | Run paired evaluation, score reference tokens, parse and score generated actions, and aggregate across draws and prefixes. |
| Utilities | [download_model.py](download_model.py), [config.py](config.py), [run_visible.ps1](run_visible.ps1) | Download model files, resolve YAML paths, and launch a Windows console. |

See [one prepared next-action example](examples/next_action.json) for the task,
prior actions, screenshot reference, and target. After preparing the data, run
`python data.py --split train --unit trajectory --index 0` to print the full messages.

## Reproduce the reported runs

The [data manifest](configs/data-manifest.json) pins the dataset revision and
row groups 0, 1, 3, and 4 from the first training shard. Data preparation uses
seed 42 to split 200 trajectories into 160 train, 20 validation, and 20 test.
After data and model setup, run the full training and evaluation configs.

~~~sh
python train.py --config configs/train-epoch.yaml
python compare_checkpoints.py --config configs/eval-nll.yaml
python compare_checkpoints.py --config configs/eval-sampled.yaml
~~~

The full run saved adapters at updates 20 and 40. The comparison writes
`action_nll.jsonl` and `nll_metrics.json` for teacher-forced likelihood, and
`predictions.jsonl` and `generation_metrics.json` for sampled actions.
See the saved [NLL report](results/nll_metrics.json),
[sampled-action report](results/generation_metrics.json), and
[technical note](NOTE.md#evaluation-and-explanation).

On Windows, `run_visible.ps1` can open the same commands in a separate
PowerShell window and keep logs under `runs/console/`.

The example comes from Ai2's MolmoWeb-HumanTrajs dataset, distributed under
ODC-BY 1.0. [Dataset and license](https://huggingface.co/datasets/allenai/MolmoWeb-HumanTrajs#license) ·
[MolmoWeb paper](https://arxiv.org/abs/2604.08516) ·
[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

AI assistance. Codex helped implement, debug, and document this pipeline.
