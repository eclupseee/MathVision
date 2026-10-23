import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.images import ImageLimits, inspect_image
from services.photo_errors import (
    CorruptImage,
    ImageResolutionTooLarge,
    ImageTooSmall,
    UnsupportedImageFormat,
)
from tests.imgutil import make_image_bytes

LIMITS = ImageLimits(max_bytes=10 * 1024 * 1024)


class InspectImageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def write(self, data: bytes) -> Path:
        path = self.dir / "incoming.bin"
        path.write_bytes(data)
        return path

    def inspect(self, data: bytes, limits: ImageLimits = LIMITS):
        path = self.write(data)
        return inspect_image(path, limits, path.stat().st_size)

    def test_valid_formats(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            with self.subTest(fmt=fmt):
                info = self.inspect(make_image_bytes(fmt, (320, 240)))
                self.assertEqual((info.format, info.width, info.height), (fmt, 320, 240))
                self.assertFalse(info.needs_rotation)

    def test_exif_orientation_swaps_dimensions(self):
        info = self.inspect(make_image_bytes("JPEG", (320, 240), orientation=6))
        self.assertEqual((info.width, info.height), (240, 320))
        self.assertTrue(info.needs_rotation)
        self.assertEqual(info.exif_orientation, 6)

    def test_exif_orientation_1_keeps_dimensions(self):
        info = self.inspect(make_image_bytes("JPEG", (320, 240), orientation=1))
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertFalse(info.needs_rotation)

    def test_truncated_jpeg_is_corrupt(self):
        data = make_image_bytes("JPEG", (400, 400))
        with self.assertRaises(CorruptImage):
            self.inspect(data[: len(data) // 2])

    def test_truncated_png_is_corrupt(self):
        data = make_image_bytes("PNG", (200, 200))
        with self.assertRaises(CorruptImage):
            self.inspect(data[: len(data) // 2])

    def test_random_bytes_are_corrupt(self):
        with self.assertRaises(CorruptImage):
            self.inspect(b"this is definitely not an image" * 20)

    def test_gif_is_unsupported(self):
        buf = io.BytesIO()
        Image.new("P", (100, 100)).save(buf, "GIF")
        with self.assertRaises(UnsupportedImageFormat):
            self.inspect(buf.getvalue())

    def test_too_many_pixels_rejected_without_decoding(self):
        buf = io.BytesIO()
        Image.new("1", (6000, 5000)).save(buf, "PNG")  # 30 МП, но файл крошечный
        with self.assertRaises(ImageResolutionTooLarge):
            self.inspect(buf.getvalue())

    def test_too_small_rejected(self):
        with self.assertRaises(ImageTooSmall):
            self.inspect(make_image_bytes("PNG", (20, 20)))


if __name__ == "__main__":
    unittest.main()
