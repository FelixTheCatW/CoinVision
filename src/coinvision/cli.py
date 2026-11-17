from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from . import __version__, analyze, data, hashes
from .config import DEFAULT_CONFIG_PATH, Config, load_config


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _load_config(args: argparse.Namespace) -> Config:
    path = Path(args.config)
    if path.is_file():
        return load_config(path)
    print(f"warning: config not found ({path}), using built-in defaults", file=sys.stderr)
    return Config()


def _require_dir(value: str) -> Path | None:
    path = Path(value)
    return path if path.is_dir() else None


def _cmd_analyze(args: argparse.Namespace, config: Config) -> int:
    directory = _require_dir(args.directory)
    if directory is None:
        return _fail(f"not a directory: {args.directory}")

    out = analyze.analyze_to_csv(
        directory,
        recursive=not args.no_recursive,
        deep_check=args.deep_check,
        progress=not args.no_progress,
    )
    print(analyze.format_report(analyze.summary_from_csv(out)))
    print(f"\nCSV -> {out}")
    return 0


def _cmd_hashes(args: argparse.Namespace, config: Config) -> int:
    directory = _require_dir(args.directory)
    if directory is None:
        return _fail(f"not a directory: {args.directory}")

    stats = hashes.hash_dataset(
        directory,
        recursive=not args.no_recursive,
        progress=not args.no_progress,
    )
    print(hashes.format_report(stats))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON -> {out}")
    return 0


def _cmd_manifest(args: argparse.Namespace, config: Config) -> int:
    source = _require_dir(args.source)
    if source is None:
        return _fail(f"not a directory: {args.source}")

    extensions = tuple(args.ext) if args.ext else config.data.extensions
    summary = data.build_manifest(
        source,
        args.out,
        extensions=extensions,
        validate=not args.no_validate,
        dedup=not args.no_dedup,
    )
    print(f"manifest -> {args.out}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


def _cmd_split(args: argparse.Namespace, config: Config) -> int:
    manifest = Path(args.manifest)
    if not manifest.is_file():
        return _fail(f"manifest not found: {manifest}")

    val_fraction = config.data.val_fraction if args.val_fraction is None else args.val_fraction
    seed = config.project.seed if args.seed is None else args.seed
    train_csv = args.train or config.data.train_manifest
    val_csv = args.val or config.data.val_manifest

    counts = data.split_manifest(manifest, train_csv, val_csv, val_fraction=val_fraction, seed=seed)
    print(f"val_fraction = {val_fraction}, seed = {seed}")
    print(f"train -> {train_csv} ({counts['train']} rows)")
    print(f"val   -> {val_csv} ({counts['val']} rows)")
    return 0


def _cmd_config(args: argparse.Namespace, config: Config) -> int:
    print(f"config file : {Path(args.config).resolve()}")
    for name in ("project", "data", "model", "train", "infer"):
        section = getattr(config, name)
        print(f"[{name}]")
        for key, value in asdict(section).items():
            print(f"  {key} = {value!r}")
    return 0


def _cmd_train(args: argparse.Namespace, config: Config) -> int:
    from . import train as train_module

    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.out_dir is not None:
        config.train.out_dir = args.out_dir

    for manifest in (config.data.train_manifest, config.data.val_manifest):
        if not Path(manifest).is_file():
            return _fail(f"manifest not found: {manifest} (run 'manifest' and 'split' first)")

    result = train_module.fit(config, device_spec=args.device)
    print(
        f"\ndevice {result['device']} | best val year accuracy "
        f"{result['best_year_accuracy']:.3f} | checkpoints -> {result['out_dir']}"
    )
    return 0


def _cmd_predict(args: argparse.Namespace, config: Config) -> int:
    from .infer import CoinPredictor

    checkpoint = Path(args.checkpoint or config.infer.checkpoint)
    if not checkpoint.is_file():
        return _fail(f"checkpoint not found: {checkpoint}")

    image = Path(args.image)
    if not image.is_file():
        return _fail(f"image not found: {image}")

    predictor = CoinPredictor.from_checkpoint(checkpoint, args.device or config.infer.device)
    result = predictor.predict(image)

    digits = " ".join(
        f"{digit}={confidence:.3f}"
        for digit, confidence in zip(
            result["digits"], result["digit_confidence"], strict=True
        )
    )
    print(f"year       : {result['year']}")
    print(f"confidence : {result['confidence']:.4f}")
    print(f"digits     : {digits}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coinvision",
        description="Coin year recognition: dataset tooling and training.",
    )
    parser.add_argument("--version", action="version", version=f"coinvision {__version__}")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"TOML config path (default: {DEFAULT_CONFIG_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    analyze_parser = subparsers.add_parser("analyze", help="scan a dataset into a CSV plus summary")
    analyze_parser.add_argument("directory")
    analyze_parser.add_argument("--no-recursive", action="store_true")
    analyze_parser.add_argument("--no-progress", action="store_true")
    deep = analyze_parser.add_mutually_exclusive_group()
    deep.add_argument(
        "--deep",
        dest="deep_check",
        action="store_true",
        default=True,
        help="full decode + pixel hashes (default)",
    )
    deep.add_argument(
        "--shallow",
        dest="deep_check",
        action="store_false",
        help="header-only check, no pixel hashes",
    )
    analyze_parser.set_defaults(func=_cmd_analyze)

    hashes_parser = subparsers.add_parser("hashes", help="hash images and report duplicates")
    hashes_parser.add_argument("directory")
    hashes_parser.add_argument("--json", default=None, help="write the full report to JSON")
    hashes_parser.add_argument("--no-recursive", action="store_true")
    hashes_parser.add_argument("--no-progress", action="store_true")
    hashes_parser.set_defaults(func=_cmd_hashes)

    manifest_parser = subparsers.add_parser("manifest", help="build a labeled manifest CSV")
    manifest_parser.add_argument("source")
    manifest_parser.add_argument("out")
    manifest_parser.add_argument("--ext", action="append", help="extension filter, repeatable")
    manifest_parser.add_argument("--no-validate", action="store_true", help="skip image checks")
    manifest_parser.add_argument("--no-dedup", action="store_true", help="keep duplicate pixels")
    manifest_parser.set_defaults(func=_cmd_manifest)

    split_parser = subparsers.add_parser("split", help="split a manifest into train/val")
    split_parser.add_argument("manifest")
    split_parser.add_argument("--train", default=None)
    split_parser.add_argument("--val", default=None)
    split_parser.add_argument("--val-fraction", type=float, default=None)
    split_parser.add_argument("--seed", type=int, default=None)
    split_parser.set_defaults(func=_cmd_split)

    config_parser = subparsers.add_parser("config", help="print the resolved config")
    config_parser.set_defaults(func=_cmd_config)

    train_parser = subparsers.add_parser("train", help="train the digit-head model")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--out-dir", default=None)
    train_parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    train_parser.set_defaults(func=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="predict the year of one coin image")
    predict_parser.add_argument("image")
    predict_parser.add_argument("--checkpoint", default=None)
    predict_parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    predict_parser.set_defaults(func=_cmd_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args, _load_config(args))


if __name__ == "__main__":
    raise SystemExit(main())
