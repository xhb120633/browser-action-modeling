"""Download the public base model into this project, recording its revision."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, snapshot_download
import requests


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_DIR = Path(__file__).parent / "models" / "Qwen2.5-VL-3B-Instruct"


def download_weight(filename, revision):
    destination = MODEL_DIR / filename
    metadata = get_hf_file_metadata(hf_hub_url(MODEL_ID, filename, revision=revision))
    if destination.exists() and destination.stat().st_size == metadata.size:
        return
    temporary = destination.with_suffix(".safetensors.partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    # Resolve by HEAD first, bypassing this older Hub client's Xet redirect path.
    with requests.get(metadata.location, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if offset:
            assert response.status_code == 206, "Server did not honor the resume range."
        with temporary.open("ab" if offset else "wb") as stream:
            for chunk in response.iter_content(8 * 1024 * 1024):
                stream.write(chunk)
    assert temporary.stat().st_size == metadata.size
    temporary.replace(destination)
    print(f"Downloaded {filename} ({metadata.size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Model commit to download. Defaults to the current Hub revision.")
    args = parser.parse_args()
    revision = args.revision or HfApi().model_info(MODEL_ID).sha
    # Download the processor first so input preparation can be checked immediately.
    snapshot_download(
        MODEL_ID,
        revision=revision,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.txt"],
        max_workers=2,
    )
    weights = [name for name in HfApi().list_repo_files(MODEL_ID, revision=revision) if name.endswith(".safetensors")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name: download_weight(name, revision), weights))
    (MODEL_DIR / "download_info.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": revision}, indent=2),
        encoding="utf-8",
    )
    print(f"Model saved to {MODEL_DIR}", flush=True)
