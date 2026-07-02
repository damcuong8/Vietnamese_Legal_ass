from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUTS = [
    BASE_DIR / "submission_eval_llama_smoke/results.json",
    BASE_DIR / "submission_eval_llama_smoke_2/results.json",
    BASE_DIR / "submission_eval_gpu0/results.json",
]
DEFAULT_OUT_DIR = BASE_DIR / "submission_eval_merged"


def load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")

    results = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: item #{idx} is not an object")
        if "id" not in item:
            raise ValueError(f"{path}: item #{idx} missing id")
        results.append(item)
    return results


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_zip(results_path: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple R2AI results.json files.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUTS,
        help="Input results.json files. Later files override earlier duplicate ids.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results-name", default="results.json")
    parser.add_argument("--zip-name", default="submission.zip")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    merged_by_id: dict[Any, dict[str, Any]] = {}
    source_by_id: dict[Any, str] = {}
    counts = []
    duplicate_count = 0

    for path in args.inputs:
        path = path.expanduser().resolve()
        results = load_results(path)
        counts.append((path, len(results)))
        for item in results:
            item_id = item["id"]
            if item_id in merged_by_id:
                duplicate_count += 1
            merged_by_id[item_id] = item
            source_by_id[item_id] = str(path)

    merged = [
        merged_by_id[item_id]
        for item_id in sorted(merged_by_id, key=lambda value: int(value))
    ]

    results_path = args.out_dir / args.results_name
    write_json(results_path, merged)

    manifest = {
        "inputs": [
            {"path": str(path), "count": count}
            for path, count in counts
        ],
        "merged_count": len(merged),
        "duplicate_overrides": duplicate_count,
        "output": str(results_path),
    }
    manifest_path = args.out_dir / "merge_manifest.json"
    write_json(manifest_path, manifest)

    if not args.no_zip:
        make_zip(results_path, args.out_dir / args.zip_name)

    print(f"Merged results: {results_path}")
    print(f"Merged count:   {len(merged)}")
    print(f"Duplicates overridden by later files: {duplicate_count}")
    print(f"Manifest:       {manifest_path}")
    if not args.no_zip:
        print(f"Submission zip: {args.out_dir / args.zip_name}")


if __name__ == "__main__":
    main()
