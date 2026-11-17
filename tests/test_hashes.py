from pathlib import Path

from PIL import Image

from coinvision.hashes import hash_dataset, hash_image, sha256_file


def _save_gradient(path: Path, size=(16, 16)):
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (x * 7 % 256, y * 5 % 256, (x + y) % 256)
    img.save(path)


def test_exact_and_content_duplicates(tmp_path: Path):
    _save_gradient(tmp_path / "a-1-1832.png")
    _save_gradient(tmp_path / "a-2-1832.png")
    _save_gradient(tmp_path / "b-1-1832.bmp")

    stats = hash_dataset(tmp_path)

    assert stats["files_hashed"] == 3
    assert stats["exact_duplicates"]["groups"] == 1
    assert stats["exact_duplicates"]["extra_files"] == 1
    assert stats["unique_file_hash"] == 2
    assert stats["content_duplicates"]["groups"] == 1
    assert stats["content_duplicates"]["extra_files"] == 2
    assert stats["unique_pixel_hash"] == 1


def test_unique_images(tmp_path: Path):
    _save_gradient(tmp_path / "x-1-1830.png", (16, 16))
    Image.new("RGB", (16, 16), (10, 20, 30)).save(tmp_path / "y-1-1831.png")

    stats = hash_dataset(tmp_path)

    assert stats["files_hashed"] == 2
    assert stats["exact_duplicates"]["groups"] == 0
    assert stats["content_duplicates"]["groups"] == 0


def test_hash_image_matches_file_hash_of_identical_files(tmp_path: Path):
    _save_gradient(tmp_path / "a.png")
    _save_gradient(tmp_path / "b.png")

    a_pixels = hash_image(tmp_path / "a.png")["pixel_sha256"]
    b_pixels = hash_image(tmp_path / "b.png")["pixel_sha256"]
    assert a_pixels == b_pixels
    assert sha256_file(tmp_path / "a.png") == sha256_file(tmp_path / "b.png")
