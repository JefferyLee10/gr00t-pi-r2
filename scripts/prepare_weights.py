"""Download the reference Hugging Face revisions or check a local model directory."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "base": ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"),
    "backbone": ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561"),
}
PROCESSOR_FILES = {
    "base": ("processor_config.json", "statistics.json", "embodiment_id.json"),
    "checkpoint": ("processor_config.json", "statistics.json", "embodiment_id.json"),
    "backbone": ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"),
}


def validate(path, kind):
    for name in ("config.json", *PROCESSOR_FILES[kind]):
        with (path / name).open(encoding="utf-8") as handle:
            json.load(handle)
    index = path / "model.safetensors.index.json"
    shards = {"model.safetensors"}
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if not mapping:
            raise ValueError(f"Empty weight map in {index}")
        shards = set(mapping.values())
    for name in sorted(shards):
        if Path(name).name != name:
            raise ValueError(f"Unexpected shard filename in {index}: {name!r}")
        # The safetensors header gives the last tensor's end offset. Checking
        # that offset detects partial downloads without loading tensors into RAM.
        shard = path / name
        with shard.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if not 2 < size <= 100_000_000:
                raise ValueError(f"Invalid safetensors header (Git LFS pointer?): {shard}")
            header = json.loads(handle.read(size))
            end = max(t["data_offsets"][1] for k, t in header.items() if k != "__metadata__")
            if shard.stat().st_size != 8 + size + end:
                raise ValueError(f"Truncated or oversized shard: {shard}")
    print(f"OK {kind}: {path} ({len(shards)} weight shard(s))")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "download"])
    parser.add_argument("kind", choices=PROCESSOR_FILES)
    parser.add_argument("--path", type=Path, help="Override the default weights directory")
    args = parser.parse_args()
    if args.kind == "checkpoint" and (args.action == "download" or args.path is None):
        parser.error("Use 'check checkpoint --path <fine-tuned checkpoint directory>'")
    path = (args.path or ROOT / "weights" / args.kind).expanduser().resolve()
    try:
        if args.action == "download":
            from huggingface_hub import snapshot_download

            repo, revision = MODELS[args.kind]
            snapshot_download(repo_id=repo, revision=revision, local_dir=path)
        validate(path, args.kind)
    except (OSError, KeyError, ValueError, ImportError) as exc:
        parser.exit(1, f"{args.kind} at {path}: {exc}\n")


if __name__ == "__main__":
    main()
