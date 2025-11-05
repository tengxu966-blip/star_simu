import unittest
from src import protocol


class TestProtocol(unittest.TestCase):
    def test_checksum(self):
        data = b'\x01\x02\x03\x04'
        self.assertEqual(protocol.calc_checksum(data), 4)

    def test_satellite_state_pack_unpack(self):
        timestamp = 123456
        pos = (1.1, 2.2, 3.3)
        vel = (4.4, 5.5, 6.6)
        quat = (0.1, 0.2, 0.3, 0.4)
        omega = (0.01, 0.02, 0.03)
        packed = protocol.pack_satellite_state(timestamp, pos, vel, quat, omega)
        self.assertEqual(len(packed), protocol.SATELLITE_STATE_TOTAL)
        self.assertEqual(protocol.calc_checksum(packed[:-1]), packed[-1])
        msg = protocol.unpack_satellite_state(packed)
        self.assertEqual(msg.timestamp, timestamp)
        self.assertAlmostEqual(msg.pos_x, pos[0], places=6)
        self.assertAlmostEqual(msg.pos_y, pos[1], places=6)
        self.assertAlmostEqual(msg.pos_z, pos[2], places=6)
        self.assertAlmostEqual(msg.vel_x, vel[0], places=6)
        self.assertAlmostEqual(msg.vel_y, vel[1], places=6)
        self.assertAlmostEqual(msg.vel_z, vel[2], places=6)
        self.assertAlmostEqual(msg.quat_x, quat[0], places=6)
        self.assertAlmostEqual(msg.quat_y, quat[1], places=6)
        self.assertAlmostEqual(msg.quat_z, quat[2], places=6)
        self.assertAlmostEqual(msg.quat_w, quat[3], places=6)
        self.assertAlmostEqual(msg.omega_x, omega[0], places=6)
        self.assertAlmostEqual(msg.omega_y, omega[1], places=6)
        self.assertAlmostEqual(msg.omega_z, omega[2], places=6)

    def test_vision_data_pack_unpack(self):
        timestamp = 654321
        u, v = 123, -456
        packed = protocol.pack_vision_data(timestamp, u, v)
        self.assertEqual(len(packed), protocol.VISION_DATA_TOTAL)
        self.assertEqual(protocol.calc_checksum(packed[:-1]), packed[-1])
        msg = protocol.unpack_vision_data(packed)
        self.assertEqual(msg.timestamp, timestamp)
        self.assertEqual(msg.u, u)
        self.assertEqual(msg.v, v)

    def test_satellite_state_checksum_error(self):
        timestamp = 1
        pos = (0.0, 0.0, 0.0)
        vel = (0.0, 0.0, 0.0)
        quat = (0.0, 0.0, 0.0, 1.0)
        omega = (0.0, 0.0, 0.0)
        packed = bytearray(protocol.pack_satellite_state(timestamp, pos, vel, quat, omega))
        packed[-1] ^= 0xFF  # 破坏校验和
        with self.assertRaises(ValueError):
            protocol.unpack_satellite_state(bytes(packed))

    def test_vision_data_checksum_error(self):
        packed = bytearray(protocol.pack_vision_data(1, 2, 3))
        packed[-1] ^= 0xFF
        with self.assertRaises(ValueError):
            protocol.unpack_vision_data(bytes(packed))

if __name__ == '__main__':
    unittest.main()
