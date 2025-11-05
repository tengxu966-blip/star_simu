# 向192.168.29.35:8888发送卫星状态数据的脚本
import socket
import struct
import time
from src.protocol import pack_satellite_state

def main():
    target_ip = '192.168.29.235'
    target_port = 8889
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(10):
        timestamp = i
        pos = (1000 + i, 0, 10000)
        vel = (0, 0, 0)
        quat = (0, 0, 0, 1)
        omega = (0, 0, 0)
        data = pack_satellite_state(timestamp, pos, vel, quat, omega)
        sock.sendto(data, (target_ip, target_port))
        print(f"[SEND] to {target_ip}:{target_port}, timestamp={timestamp}")
        time.sleep(0.5)
    sock.close()

if __name__ == '__main__':
    main()
