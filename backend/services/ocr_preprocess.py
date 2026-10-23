"""Подготовка изображения к распознаванию (синхронно; вызывать из потока)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

MIN_WIDTH = 1200  # мелкий текст на маленьких картинках распознаётся хуже
MAX_SIDE = 4000
MAX_UPSCALE = 3.0


def prepare_image_for_ocr(src: Path, dst: Path) -> None:
    """EXIF-поворот, серый цвет, автоконтраст, разумный масштаб -> PNG в dst."""
    with Image.open(src) as opened:
        if opened.format == "JPEG":  # дешёвое уменьшение при декодировании
            opened.draft(None, (MAX_SIDE, MAX_SIDE))
        img = ImageOps.exif_transpose(opened)
        img.load()

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        img = Image.alpha_composite(background, rgba)
    img = ImageOps.autocontrast(img.convert("L"), cutoff=1)

    width, height = img.size
    longest = max(width, height)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
    elif width < MIN_WIDTH:
        scale = min(MAX_UPSCALE, MIN_WIDTH / width)
    else:
        scale = 1.0
    if scale != 1.0:
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    img.save(dst, "PNG")
