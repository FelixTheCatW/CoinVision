from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .analyze import EXTENSION_FORMATS, sha256_file

DEFAULT_EXTENSIONS = tuple(EXTENSION_FORMATS)


def hash_image(path: str | Path) -> dict:
    try:
        with Image.open(path) as img:
            img.load()
            rgb = img.convert("RGB")
            width, height = rgb.size
            pixels = rgb.tobytes()
    except Exception as exc:  # noqa: BLE001 - report any decode failure
        return {"ok": False, "detail": str(exc)}

    return {
        "ok": True,
        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
        "width": width,
        "height": height,
    }


def _groups_from_map(mapping: dict[str, list[int]], paths: list[str]) -> list[list[str]]:
    groups = [sorted(paths[i] for i in ids) for ids in mapping.values() if len(ids) > 1]
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def hash_dataset(
    directory: str | Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    recursive: bool = True,
    max_examples: int = 20,
    progress: bool = True,
) -> dict:
    directory = Path(directory)
    exts = tuple((e if e.startswith(".") else f".{e}").lower() for e in extensions)
    bars_off = not progress or not sys.stderr.isatty()

    entries: list[dict] = []
    broken: list[dict] = []
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    scan = tqdm(iterator, desc="[1/2] scan files", unit=" file", disable=bars_off)
    image_paths = sorted(
        path for path in scan if path.is_file() and path.suffix.lower() in exts
    )
    scan.close()

    hashing = tqdm(image_paths, desc="[2/2] hash images", unit=" img", disable=bars_off)
    for path in hashing:
        result = hash_image(path)
        if not result["ok"]:
            broken.append({"path": str(path.as_posix()), "detail": result["detail"]})
            continue
        entries.append(
            {
                "path": str(path.as_posix()),
                "file_sha256": sha256_file(path),
                "pixel_sha256": result["pixel_sha256"],
                "width": result["width"],
                "height": result["height"],
            }
        )
    hashing.close()

    paths = [entry["path"] for entry in entries]
    by_file: dict[str, list[int]] = defaultdict(list)
    by_pixel: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        by_file[entry["file_sha256"]].append(index)
        by_pixel[entry["pixel_sha256"]].append(index)

    file_groups = _groups_from_map(by_file, paths)
    pixel_groups = _groups_from_map(by_pixel, paths)

    def _examples(index: dict[str, list[int]], key: str) -> list[dict]:
        out: list[dict] = []
        for sha, ids in sorted(index.items(), key=lambda kv: -len(kv[1])):
            if len(ids) > 1:
                out.append({key: sha, "count": len(ids), "paths": sorted(paths[i] for i in ids)})
            if len(out) >= max_examples:
                break
        return out

    return {
        "directory": str(directory.as_posix()),
        "recursive": recursive,
        "hash": {"file": "sha256(bytes)", "pixel": "sha256(rgb_tobytes)"},
        "files_hashed": len(entries),
        "broken": {"count": len(broken), "items": broken},
        "unique_file_hash": len(by_file),
        "unique_pixel_hash": len(by_pixel),
        "exact_duplicates": {
            "groups": len(file_groups),
            "extra_files": sum(len(g) - 1 for g in file_groups),
            "examples": _examples(by_file, "sha256"),
        },
        "content_duplicates": {
            "groups": len(pixel_groups),
            "extra_files": sum(len(g) - 1 for g in pixel_groups),
            "examples": _examples(by_pixel, "pixel_sha256"),
        },
        "items": entries if len(entries) <= 5000 else [],
    }


def format_report(stats: dict) -> str:
    ex = stats["exact_duplicates"]
    ct = stats["content_duplicates"]
    lines = [
        f"Directory        : {stats['directory']}",
        f"Hashes           : {stats['hash']}",
        f"Files hashed     : {stats['files_hashed']}",
        f"Broken           : {stats['broken']['count']}",
        f"Unique by file   : {stats['unique_file_hash']}",
        f"Unique by pixels : {stats['unique_pixel_hash']}",
        f"Exact duplicates : {ex['groups']} groups, {ex['extra_files']} extra files",
        f"Content dupes    : {ct['groups']} groups, {ct['extra_files']} extra files",
    ]
    return "\n".join(lines)


def main(directory: str, json_out: str | None = None) -> dict:
    stats = hash_dataset(directory)
    print(format_report(stats))
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON -> {json_out}")
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hash a coin image dataset and check uniqueness")
    parser.add_argument("directory")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()
    main(args.directory, args.json_out)
