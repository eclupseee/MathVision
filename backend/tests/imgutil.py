"""Вспомогательные генераторы изображений для тестов (без сети и файлов в репозитории)."""
from __future__ import annotations

import io
import os

from PIL import Image


def make_image_bytes(fmt="JPEG", size=(320, 240), orientation=None, noise=True) -> bytes:
    if noise:  # шум плохо сжимается -> файл «настоящего» размера, обрезка ломает данные
        img = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    else:
        img = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()
