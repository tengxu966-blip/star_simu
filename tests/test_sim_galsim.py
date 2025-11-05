import unittest
from src import sim_galsim, attitude
import numpy as np
import os
from datetime import datetime
from PIL import Image

class DummySatMsg:
    def __init__(self, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.quat_x = quat_x
        self.quat_y = quat_y
        self.quat_z = quat_z
        self.quat_w = quat_w

class TestSimGalsim(unittest.TestCase):
    def setUp(self):
        self.debug_dir = 'debug_output'
        os.makedirs(self.debug_dir, exist_ok=True)
        self.logfile = os.path.join(self.debug_dir, 'test_log.txt')
        with open(self.logfile, 'a') as f:
            f.write(f"\n--- Test started at {datetime.now()} ---\n")

    def test_generate_asteroid_image_basic(self):
        sat = DummySatMsg(1000, 0, 10000, 0, 0, 0, 1)
        img = sim_galsim.generate_asteroid_image(sat, gaia_catalog_path=None)
        self.assertEqual(img.shape, (256, 256))
        self.assertTrue(np.any(img > 0))
        # 保存图片
        img_path = os.path.join(self.debug_dir, 'asteroid_test.png')
        Image.fromarray(img).save(img_path)
        # 保存日志
        with open(self.logfile, 'a') as f:
            f.write(f"Saved asteroid image to {img_path}\n")

    def test_attitude_center(self):
        quat = (0, 0, 0, 1)
        ra, dec = attitude.get_fov_center_ra_dec(None, quat)
        self.assertTrue(0 <= ra < 360)
        self.assertTrue(-90 <= dec <= 90)
        # 保存日志
        with open(self.logfile, 'a') as f:
            f.write(f"Attitude center: ra={ra}, dec={dec}\n")

if __name__ == '__main__':
    unittest.main()
