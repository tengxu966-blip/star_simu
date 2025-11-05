# 向192.168.29.35:8888发送视觉数据包的脚本
import socket
import struct
import time
from src.protocol import pack_vision_data

def main():
    target_ip = '192.168.29.235'
    target_port = 8889
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(1000):
        timestamp = i
        u = 100 + i
        v = 200 + i
        confidence = 0.8
        data = pack_vision_data(timestamp, u, v, confidence)
        sock.sendto(data, (target_ip, target_port))
        print(f"[SEND] VisionData to {target_ip}:{target_port}, t={timestamp}, u={u}, v={v}, conf={confidence}")
        time.sleep(0.5)
    sock.close()

if __name__ == '__main__':
    main()
