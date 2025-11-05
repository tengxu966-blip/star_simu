"""
协议字段用proto管理，打包/解包仍用struct，严格按原定字节格式
"""

import socket
import struct
from typing import Tuple
from src import protocol_pb2
import toml
import os

# 全局缓存配置
_sim_config = None
def get_sim_config():
    global _sim_config
    if _sim_config is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sim_config.toml')
        if os.path.exists(config_path):
            _sim_config = toml.load(config_path)
        else:
            _sim_config = {}
    return _sim_config

# 卫星状态数据包格式（81字节）
SATELLITE_STATE_FORMAT = '<I3d3d4f3f'  # 不含校验和
SATELLITE_STATE_SIZE = struct.calcsize(SATELLITE_STATE_FORMAT)  # 80
SATELLITE_STATE_TOTAL = 81

 # 视觉数据包格式（13字节）
VISION_DATA_FORMAT = '<Ihhf'  # 不含校验和
VISION_DATA_SIZE = struct.calcsize(VISION_DATA_FORMAT)  # 12
VISION_DATA_TOTAL = 13

def calc_checksum(data: bytes) -> int:
    """计算校验和（所有字节求和后对256取余）"""
    checksum = sum(data) % 256
    return checksum

class UdpComm:
    def __init__(self, recv_port: int, send_addr: Tuple[str, int]):
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_recv.bind(('0.0.0.0', recv_port))
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_addr = send_addr

    def recv(self, bufsize=1024) -> Tuple[bytes, Tuple[str, int]]:
        return self.sock_recv.recvfrom(bufsize)

    def send(self, data: bytes):
        self.sock_send.sendto(data, self.send_addr)

# 卫星状态数据打包/解包（字段顺序和类型参考proto）
def pack_satellite_state(timestamp: int, pos, vel, quat, omega) -> bytes:
    data = struct.pack(
        SATELLITE_STATE_FORMAT,
        timestamp,
        *pos,
        *vel,
        *quat,
        *omega
    )
    config = get_sim_config()
    if config.get('enable_checksum', True):
        checksum = calc_checksum(data)
        return data + struct.pack('B', checksum)
    else:
        return data + b'\x00'

def unpack_satellite_state(data: bytes):
    if len(data) != SATELLITE_STATE_TOTAL:
        raise ValueError('Invalid satellite state packet size')
    config = get_sim_config()
    if config.get('enable_checksum', True):
        # 可选：如需校验可在此处加校验和判断
        pass
    fields = struct.unpack(SATELLITE_STATE_FORMAT, data[:-1])
    msg = protocol_pb2.SatelliteState(
        timestamp=int(fields[0]),
        pos_x=fields[1], pos_y=fields[2], pos_z=fields[3],
        vel_x=fields[4], vel_y=fields[5], vel_z=fields[6],
        quat_x=fields[7], quat_y=fields[8], quat_z=fields[9], quat_w=fields[10],
        omega_x=fields[11], omega_y=fields[12], omega_z=fields[13]
    )
    return msg

# 视觉数据打包/解包
def pack_vision_data(timestamp: int, u: int, v: int, confidence: float) -> bytes:
    data = struct.pack(VISION_DATA_FORMAT, timestamp, u, v, confidence)
    config = get_sim_config()
    if config.get('enable_checksum', True):
        checksum = calc_checksum(data)
        return data + struct.pack('B', checksum)
    else:
        return data + b'\x00'

def unpack_vision_data(data: bytes):
    print('data is ', data)
    if len(data) != VISION_DATA_TOTAL:
        raise ValueError('Invalid vision data packet size')
    config = get_sim_config()
    if config.get('enable_checksum', True):
        # 可选：如需校验可在此处加校验和判断
        pass
    fields = struct.unpack(VISION_DATA_FORMAT, data[:-1])
    msg = protocol_pb2.VisionData(
        timestamp=int(fields[0]),
        u=int(fields[1]),
        v=int(fields[2]),
        confidence=float(fields[3])
    )
    return msg
