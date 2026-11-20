from pathlib import Path

from coinvision.config import Config, DataConfig, load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_sections_missing(tmp_path: Path):
    config = load_config(_write(tmp_path, ""), base=None)

    assert isinstance(config.data, DataConfig)
    assert config.data.image_size == (448, 224)
    assert config.data.extensions[0] == ".webp"
    assert config.train.epochs == 30
    assert config.infer.device == "auto"
    assert config.project.seed == 42


def test_partial_config_inherits_the_base_file(tmp_path: Path):
    base = tmp_path / "base.toml"
    base.write_text('[model]\npretrained = true\nbackbone = "resnet18"\n', encoding="utf-8")
    experiment = tmp_path / "experiment.toml"
    experiment.write_text("[train]\nepochs = 60\n", encoding="utf-8")

    config = load_config(experiment, base=base)

    assert config.train.epochs == 60
    # the whole point: an experiment file must not silently drop the base settings
    assert config.model.pretrained is True
    assert config.train.lr == 0.001


def test_project_experiment_keeps_pretrained(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "exp_epochs60.toml")

    assert config.train.epochs == 60
    assert config.model.pretrained is True
    assert config.data.cache is True


def test_toml_overrides_and_tuple_coercion(tmp_path: Path):
    path = _write(
        tmp_path,
        """
[project]
seed = 7

[data]
image_size = [320, 160]
extensions = [".png"]
val_fraction = 0.25

[train]
epochs = 3
batch_size = 4
""",
    )

    config = load_config(path, base=None)

    assert config.project.seed == 7
    assert config.data.image_size == (320, 160)
    assert all(side > 0 for side in config.data.image_size)
    assert config.data.extensions == (".png",)
    assert config.data.val_fraction == 0.25
    assert config.train.epochs == 3
    assert config.train.batch_size == 4
    assert config.train.lr == 0.001


def test_unknown_keys_and_sections_ignored(tmp_path: Path):
    path = _write(tmp_path, "[train]\nepochs = 5\nlegacy_option = true\n\n[unknown]\nx = 1\n")

    config = load_config(path, base=None)

    assert config.train.epochs == 5
    assert not hasattr(config, "unknown")


def test_project_config_file(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.toml")

    assert isinstance(config, Config)
    assert config.model.backbone == "resnet18"
    # ImageNet init beats from-scratch training on this dataset by a wide margin.
    assert config.model.pretrained is True
    # The 5-fold winner: augmentation, 60 epochs, label smoothing and a warmup+cosine schedule.
    assert config.data.augment is True
    assert config.train.epochs == 60
    assert config.train.label_smoothing == 0.1
    assert config.train.lr_schedule == "cosine"
    assert config.model.num_classes_per_digit == 10
    assert config.train.out_dir == "runs"
