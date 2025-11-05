"""
视觉端主程序示例
"""
from src import protocol
import time

def vision_process(sat_msg):
    """
    视觉处理逻辑，输入为SatelliteState消息，输出像素坐标(u, v)
    这里用虚拟数据，实际可替换为视觉算法
    """
    u = int(sat_msg.pos_x * 10) % 640
    v = int(sat_msg.pos_y * 10) % 480
    confidence = 0.9
    return u, v, confidence

def main():
    # 配置
    RECV_PORT = 8888
    SEND_ADDR = ('192.168.1.10', 8889)
    comm = protocol.UdpComm(RECV_PORT, SEND_ADDR)
    print(f"Listening on UDP port {RECV_PORT}...")
    while True:
        data, addr = comm.recv()
        try:
            sat_msg = protocol.unpack_satellite_state(data)
            print(f"Received state: timestamp={sat_msg.timestamp}, pos=({sat_msg.pos_x:.2f},{sat_msg.pos_y:.2f},{sat_msg.pos_z:.2f})")
            # 视觉处理
            u, v, confidence = vision_process(sat_msg)
            vision_pack = protocol.pack_vision_data(sat_msg.timestamp, u, v, confidence)
            comm.send(vision_pack)
            print(f"Sent vision: timestamp={sat_msg.timestamp}, u={u}, v={v}, confidence={confidence:.2f}")
        except Exception as e:
            print(f"Packet error: {e}")
        time.sleep(0.01)

if __name__ == '__main__':
    main()
