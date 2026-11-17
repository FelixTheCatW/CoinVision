from pathlib import Path

from coinvision.config import load_config
from coinvision.utils import parse_year


def test_parse_year_from_filename():
    assert parse_year("data/source/68891b440c55205080b14143-2-1832.webp") == "1832"
    assert parse_year("646c51902e7e293b0787acb6-1-1833.jpg") == "1833"


def test_parse_year_missing():
    assert parse_year("some/coin-without-label.webp") is None


def test_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.toml")
    assert len(config.data.image_size) == 2
    assert all(side > 0 for side in config.data.image_size)
    assert 0 < config.data.val_fraction < 1
    assert config.model.num_digits == 4
