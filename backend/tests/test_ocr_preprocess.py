import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.ocr_preprocess import MAX_SIDE, MIN_WIDTH, prepare_image_for_ocr
from tests.imgutil import make_image_bytes


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def run_prepare(self, data: bytes) -> Image.Image:
        src, dst = self.dir / "in.bin", self.dir / "out.png"
        src.write_bytes(data)
        prepare_image_for_ocr(src, dst)
        with Image.open(dst) as out:
            out.load()
            self.assertEqual(out.format, "PNG")
            self.assertEqual(out.mode, "L")
            return out.copy()

    def test_small_image_is_upscaled(self):
        out = self.run_prepare(make_image_bytes("PNG", (400, 200)))
        self.assertGreaterEqual(out.width, MIN_WIDTH)

    def test_big_image_is_downscaled(self):
        out = self.run_prepare(make_image_bytes("PNG", (5000, 100), noise=False))
        self.assertEqual(max(out.size), MAX_SIDE)

    def test_exif_orientation_applied(self):
        out = self.run_prepare(make_image_bytes("JPEG", (1300, 600), orientation=6))
        self.assertGreater(out.height, out.width)  # повернули на 90°

    def test_transparent_png_gets_white_background(self):
        buf = self.dir / "t.png"
        Image.new("RGBA", (1300, 300), (0, 0, 0, 0)).save(buf)
        out = self.run_prepare(buf.read_bytes())
        self.assertGreater(out.getpixel((5, 5)), 240)  # прозрачное -> белое, не чёрное


if __name__ == "__main__":
    unittest.main()
