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

    seed = config.project.seed if args.seed is None else args.seed
    if args.folds is not None:
        out_dir = args.out_dir or str(Path(config.data.manifest_dir) / f"folds{args.folds}")
        fold_counts = data.make_folds(
            manifest,
            out_dir,
            folds=args.folds,
            seed=seed,
            group_by_listing=not args.no_group_listing,
        )
        print(f"folds = {args.folds}, seed = {seed}, grouping = {not args.no_group_listing}")
        for name, counts_per_fold in fold_counts.items():
            print(
                f"{name}: train {counts_per_fold['train']} rows, "
                f"val {counts_per_fold['val']} rows"
            )
        print(f"manifests -> {out_dir}")
        return 0

    val_fraction = config.data.val_fraction if args.val_fraction is None else args.val_fraction
    train_csv = args.train or config.data.train_manifest
    val_csv = args.val or config.data.val_manifest

    counts = data.split_manifest(
        manifest,
        train_csv,
        val_csv,
        val_fraction=val_fraction,
        seed=seed,
        group_by_listing=args.group_listing,
    )
    print(
        f"val_fraction = {val_fraction}, seed = {seed}, "
        f"grouping = {args.group_listing}"
    )
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
    if args.train_manifest is not None:
        config.data.train_manifest = args.train_manifest
    if args.val_manifest is not None:
        config.data.val_manifest = args.val_manifest
    if args.no_cache:
        config.data.cache = False
    if args.deterministic:
        config.train.deterministic = True
    if args.seed is not None:
        config.project.seed = args.seed

    for manifest in (config.data.train_manifest, config.data.val_manifest):
        if not Path(manifest).is_file():
            return _fail(f"manifest not found: {manifest} (run 'manifest' and 'split' first)")

    result = train_module.fit(config, device_spec=args.device)
    report = result["val_last_digit"]
    print(
        f"\ndevice {result['device']} | best epoch {result['best_epoch']} | "
        f"last-digit acc {report['accuracy']:.3f} (majority baseline "
        f"{report['majority_baseline']:.3f}) | macro F1 {report['macro_f1']:.3f} | "
        f"checkpoints -> {result['out_dir']}"
    )
    return 0


def _cmd_export(args: argparse.Namespace, config: Config) -> int:
    from . import export

    for checkpoint in args.checkpoint:
        if not Path(checkpoint).is_file():
            return _fail(f"checkpoint not found: {checkpoint}")

    image_size = (int(config.data.image_size[0]), int(config.data.image_size[1]))
    metadata = export.export_onnx(
        args.checkpoint,
        args.out_dir,
        image_size=image_size,
        opset=args.opset,
        verify=not args.no_verify,
    )
    print(f"exported {len(metadata['models'])} model(s) to {args.out_dir}")
    for entry in metadata["models"]:
        diff = entry.get("max_abs_diff")
        suffix = f", max|torch-onnx| = {diff}" if diff is not None else ""
        print(f"  {entry['file']} ({int(entry['size_bytes']) / 1024**2:.1f} MiB){suffix}")
    print(f"metadata -> {Path(args.out_dir) / export.METADATA_FILENAME}")
    return 0


def _cmd_ensemble(args: argparse.Namespace, config: Config) -> int:
    from . import ensemble
    from .train import format_last_digit_report

    manifest = Path(args.manifest)
    if not manifest.is_file():
        return _fail(f"manifest not found: {manifest}")
    for checkpoint in args.checkpoint:
        if not Path(checkpoint).is_file():
            return _fail(f"checkpoint not found: {checkpoint}")

    image_size = (int(config.data.image_size[0]), int(config.data.image_size[1]))
    report = ensemble.evaluate_ensemble(
        args.checkpoint,
        manifest,
        image_size,
        views=args.tta,
        device_spec=args.device or config.infer.device,
        num_classes=config.model.num_classes_per_digit,
    )
    print(format_last_digit_report(report))
    print(f"\ncheckpoints: {len(args.checkpoint)}, views: {args.tta}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON -> {out}")
    return 0


def _cmd_plot(args: argparse.Namespace, config: Config) -> int:
    from . import plots

    run_dir = Path(args.run_dir)
    if not (run_dir / plots.METRICS_FILENAME).is_file():
        return _fail(f"metrics.json not found in {run_dir} (train a model first)")

    paths = plots.render_run(run_dir)
    print(f"run -> {run_dir}")
    for name, path in paths.items():
        print(f"  {name} -> {path}")
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


def _add_config_option(parser: argparse.ArgumentParser, *, suppress_default: bool = False) -> None:
    """Register --config.

    The flag is accepted both before and after the subcommand: subparsers register it with a
    SUPPRESS default so an earlier value is not overwritten by argparse's own default.
    """
    parser.add_argument(
        "--config",
        default=argparse.SUPPRESS if suppress_default else str(DEFAULT_CONFIG_PATH),
        help=f"TOML config path (default: {DEFAULT_CONFIG_PATH})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coinvision",
        description="Coin year recognition: dataset tooling and training.",
    )
    parser.add_argument("--version", action="version", version=f"coinvision {__version__}")
    _add_config_option(parser)
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
    split_parser.add_argument(
        "--group-listing",
        action="store_true",
        help="keep every shot of one listing inside a single split",
    )
    split_parser.add_argument(
        "--folds", type=int, default=None, help="write this many stratified fold manifests"
    )
    split_parser.add_argument(
        "--out-dir", default=None, help="where fold manifests go (default: data/manifests/foldsN)"
    )
    split_parser.add_argument(
        "--no-group-listing",
        action="store_true",
        help="with --folds: ignore listing groups (leaky, for comparison only)",
    )
    split_parser.set_defaults(func=_cmd_split)

    config_parser = subparsers.add_parser("config", help="print the resolved config")
    config_parser.set_defaults(func=_cmd_config)

    train_parser = subparsers.add_parser("train", help="train the digit-head model")
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--out-dir", default=None)
    train_parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    train_parser.add_argument(
        "--train-manifest", default=None, help="override data.train_manifest (e.g. a CV fold)"
    )
    train_parser.add_argument(
        "--val-manifest", default=None, help="override data.val_manifest (e.g. a CV fold)"
    )
    train_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="decode images every epoch instead of using the in-memory pixel cache",
    )
    train_parser.add_argument(
        "--deterministic",
        action="store_true",
        help="bit-reproducible CUDA kernels (slower, but two runs give the same numbers)",
    )
    train_parser.add_argument(
        "--seed", type=int, default=None, help="override project.seed (multi-seed ensembles)"
    )
    train_parser.set_defaults(func=_cmd_train)

    predict_parser = subparsers.add_parser("predict", help="predict the year of one coin image")
    predict_parser.add_argument("image")
    predict_parser.add_argument("--checkpoint", default=None)
    predict_parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    predict_parser.set_defaults(func=_cmd_predict)

    plot_parser = subparsers.add_parser(
        "plot", help="re-render the history/confusion plots of a run from metrics.json"
    )
    plot_parser.add_argument("run_dir", help="run directory that holds metrics.json")
    plot_parser.set_defaults(func=_cmd_plot)

    ensemble_parser = subparsers.add_parser(
        "ensemble", help="evaluate the average of several checkpoints over one or more views"
    )
    ensemble_parser.add_argument("--manifest", required=True)
    ensemble_parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="checkpoint path, repeatable",
    )
    ensemble_parser.add_argument("--tta", type=int, default=1, help="number of views (1-3)")
    ensemble_parser.add_argument("--device", default=None, help="auto / cuda / cpu")
    ensemble_parser.add_argument("--json", default=None)
    ensemble_parser.set_defaults(func=_cmd_ensemble)

    export_parser = subparsers.add_parser(
        "export", help="export checkpoints to ONNX with a preprocessing metadata sidecar"
    )
    export_parser.add_argument(
        "--checkpoint", action="append", required=True, help="checkpoint path, repeatable"
    )
    export_parser.add_argument("--out-dir", default="onnx", help="output directory")
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.add_argument(
        "--no-verify", action="store_true", help="skip the onnxruntime comparison"
    )
    export_parser.set_defaults(func=_cmd_export)

    # `coinvision train --config x.toml` should work just like `coinvision --config x.toml train`.
    for subparser in subparsers.choices.values():
        _add_config_option(subparser, suppress_default=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args, _load_config(args))


if __name__ == "__main__":
    raise SystemExit(main())
