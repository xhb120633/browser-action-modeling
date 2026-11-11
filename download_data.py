"""Download a small, reproducible subset without fetching whole 9 GB shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import fsspec
from huggingface_hub import HfApi, HfFileSystem, get_hf_file_metadata, hf_hub_url
import pyarrow.parquet as pq


REPO = "allenai/MolmoWeb-HumanTrajs"
DATA_DIR = Path(__file__).parent / "data"


def download_part(part, url, output_dir):
    destination = output_dir / part["file"]
    if destination.exists():
        assert pq.read_metadata(destination).num_rows == part["trajectories"]
        return part["trajectories"]
    with fsspec.open(url, "rb", block_size=8 * 1024 * 1024,
                     client_kwargs={"trust_env": True}) as stream:
        table = pq.ParquetFile(stream).read_row_group(part["row_group"])
        table = table.slice(0, part["trajectories"])
        temporary = destination.with_suffix(".parquet.partial")
        pq.write_table(table, temporary)
        temporary.replace(destination)
    return table.num_rows


def first_trajectories(revision, count):
    fs = HfFileSystem()
    source_root = f"datasets/{REPO}@{revision}"
    source_files = sorted(fs.glob(f"{source_root}/data/train-*.parquet"))
    selected = 0
    parts = []

    for source_file in source_files:
        # Inspect the footer first; no screenshots are downloaded here.
        with fs.open(source_file, "rb", block_size=1024 * 1024) as stream:
            parquet = pq.ParquetFile(stream)
            for group_index in range(parquet.num_row_groups):
                rows = min(count - selected, parquet.metadata.row_group(group_index).num_rows)
                filename = f"part-{len(parts):05d}.parquet"
                parts.append({
                    "file": filename,
                    "source_file": source_file,
                    "row_group": group_index,
                    "trajectories": rows,
                })
                selected += rows
                if selected == count:
                    break
        if selected == count:
            break

    if selected != count:
        raise ValueError(f"Requested {count} trajectories, found {selected}.")

    return {
        "dataset": REPO,
        "revision": revision,
        "selection": "first N trajectories in sorted train shards; not a random sample",
        "trajectories": selected,
        "parts": parts,
    }


def download_trajectories(output_dir: Path, count: int = 200, *, source_manifest: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.json"
    plan_path = output_dir / "download_plan.json"
    revision_path = output_dir / "revision.txt"
    requested = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest else None
    if requested:
        count = requested["trajectories"]

    if manifest_path.exists() or plan_path.exists():
        recorded_path = manifest_path if manifest_path.exists() else plan_path
        manifest = json.loads(recorded_path.read_text(encoding="utf-8"))
        subset_fields = ("dataset", "revision", "trajectories", "parts")
        if manifest["trajectories"] != count or (
            requested and any(manifest[key] != requested[key] for key in subset_fields)
        ):
            raise ValueError("Use a new output directory for a different subset.")
    elif requested:
        if any(output_dir.glob("part-*.parquet")):
            raise ValueError("Use a new output directory for a pinned subset without a recorded plan.")
        manifest = requested
    else:
        revision = revision_path.read_text().strip() if revision_path.exists() else HfApi().dataset_info(REPO).sha
        manifest = first_trajectories(revision, count)

    revision = manifest["revision"]
    parts = manifest["parts"]
    names = {part["file"] for part in parts}
    if manifest["dataset"] != REPO or len(names) != len(parts) or any(
        name != Path(name).name or not name.startswith("part-") or not name.endswith(".parquet") for name in names
    ) or sum(part["trajectories"] for part in parts) != count:
        raise ValueError("Invalid subset manifest.")
    existing = {file.name for file in output_dir.glob("part-*.parquet")}
    if existing - names:
        raise ValueError("Use a new output directory for a different subset.")
    if manifest_path.exists() and existing == names:
        print(f"Reusing {count} downloaded trajectories in {output_dir}", flush=True)
        return

    # Record the selection before downloading so retries use the same source rows.
    revision_path.write_text(revision, encoding="utf-8")
    plan_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    urls = {}
    for source_file in {part["source_file"] for part in parts}:
        url = hf_hub_url(REPO, f"data/{Path(source_file).name}", repo_type="dataset", revision=revision)
        urls[source_file] = get_hf_file_metadata(url).location

    # A few independent range downloads avoid waiting on one large response.
    downloaded = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(download_part, part, urls[part["source_file"]], output_dir): part
            for part in parts
        }
        for future in as_completed(futures):
            downloaded += future.result()
            print(f"Available {downloaded}/{count} trajectories: {futures[future]['file']}", flush=True)
    if downloaded != count:
        raise ValueError(f"Requested {count} trajectories, downloaded {downloaded}.")
    plan_path.replace(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--count", type=int, default=200)
    subset.add_argument("--manifest", type=Path, help="Download the exact row groups in this manifest.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "raw")
    args = parser.parse_args()
    download_trajectories(args.output_dir, args.count, source_manifest=args.manifest)
