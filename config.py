"""Read YAML and resolve local paths relative to the configuration file."""

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, key in (
        ("model", "path"), ("data", "directory"), ("training", "output_dir"),
        ("inference", "adapter_path"), ("evaluation", "output_dir"),
    ):
        value = config.get(section, {}).get(key)
        if value is not None:
            config[section][key] = str((path.parent / value).resolve())
    if "checkpoints" in config.get("evaluation", {}):
        config["evaluation"]["checkpoints"] = {
            name: str((path.parent / value).resolve())
            for name, value in config["evaluation"]["checkpoints"].items()
        }
    return config


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
