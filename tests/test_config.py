from pathlib import Path

from coinvision.config import Config, DataConfig, load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_sections_missing(tmp_path: Path):
    config = load_config(_write(tmp_path, ""))

    assert isinstance(config.data, DataConfig)
    assert config.data.image_size == (448, 224)
    assert config.data.extensions[0] == ".webp"
    assert config.train.epochs == 30
    assert config.infer.device == "auto"
    assert config.project.seed == 42


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

    config = load_config(path)

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

    config = load_config(path)

    assert config.train.epochs == 5
    assert not hasattr(config, "unknown")


def test_project_config_file(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.toml")

    assert isinstance(config, Config)
    assert config.model.backbone == "resnet18"
    assert config.model.pretrained is False
    assert config.model.num_classes_per_digit == 10
    assert config.train.out_dir == "runs"
