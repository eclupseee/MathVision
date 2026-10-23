from dataclasses import dataclass

from services.validation import check_photo_size, pick_largest_photo


@dataclass
class FakePhoto:
    width: int
    height: int
    file_size: int | None = None


def test_pick_largest():
    small, big = FakePhoto(90, 90, 1000), FakePhoto(1280, 960, 200000)
    assert pick_largest_photo([small, big]) is big


def test_pick_largest_empty():
    assert pick_largest_photo([]) is None


def test_size_ok_and_too_big_and_unknown():
    limit = 1024
    assert check_photo_size(1024, limit).ok
    too_big = check_photo_size(1025, limit)
    assert not too_big.ok and too_big.reason
    assert check_photo_size(None, limit).ok
