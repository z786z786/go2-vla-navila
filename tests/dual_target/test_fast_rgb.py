import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.dual_target.fast_rgb import AsyncRgbWriter

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


@unittest.skipIf(np is None or cv2 is None, 'requires numpy/OpenCV')
class WriterTests(unittest.TestCase):
    def test_snapshot_bounded_and_exact(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);writer=AsyncRgbWriter(pending_limit=2)
            expected={}
            for i in range(12):
                rgb=np.full((16,16,3),i,dtype=np.uint8);rgb[:,:,0]=200
                path=root/f'{i}.png';expected[path]=rgb.copy();writer(path,rgb)
                rgb[:]=0
                self.assertLessEqual(len(writer.pending),2)
            writer.close()
            for path,rgb in expected.items():
                self.assertTrue(np.array_equal(cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB),rgb))

    def test_reset_synchronous_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rgb_reset';p.mkdir();p=p/'0.png';w=AsyncRgbWriter()
            try:
                w(p,np.zeros((8,8,3),dtype=np.uint8));self.assertTrue(p.exists())
                with self.assertRaises(ValueError):w(p,np.zeros((8,8,3),dtype=np.uint8))
            finally:w.close()

    def test_failure_propagates_at_close(self):
        with tempfile.TemporaryDirectory() as d:
            w=AsyncRgbWriter()
            with patch('cv2.imwrite',return_value=False):
                w(Path(d)/'0.png',np.zeros((8,8,3),dtype=np.uint8))
                with self.assertRaises(OSError):w.close()

    def test_closed_and_invalid_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w=AsyncRgbWriter();p=Path(d)/'a.png'
            with self.assertRaises(ValueError):w(p,np.zeros((2,2),dtype=np.uint8))
            w.close()
            with self.assertRaises(ValueError):w(p,np.zeros((2,2,3),dtype=np.uint8))


if __name__=='__main__':unittest.main()
