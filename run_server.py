"""
持续监听UDP服务器脚本
- 仅启动GNC异步服务器，持续监听MATLAB端数据
- 不自动退出
"""
import asyncio
from src.async_server import GNCServerProtocol

async def main():
    loop = asyncio.get_running_loop()
    # 启动GNC异步服务器，监听8888端口
    await loop.create_datagram_endpoint(
        lambda: GNCServerProtocol(), local_addr=('0.0.0.0', 8888))
    print("[INFO] GNC UDP服务器已启动，持续监听中...")

    # 启动MATLAB模拟接收端（返回数据给matlab）

    class MatlabReceiver(asyncio.DatagramProtocol):
        def datagram_received(self, data, addr):
            print(f"[MATLAB] Got vision: raw={data}")
            from src.protocol import unpack_vision_data
            try:
                msg = unpack_vision_data(data)
                print(f"[MATLAB] Got vision: t={msg.timestamp}, u={msg.u}, v={msg.v}, conf={msg.confidence:.3f}")
            except Exception as e:
                print(f"[MATLAB] unpack_vision_data failed: {e}")
            # 转发数据到指定IP
            target_ip = '192.168.29.235'
            target_port = 8889
            sock = asyncio.DatagramTransport
            # 直接用loop发送UDP包
            loop = asyncio.get_event_loop()
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            send_sock.sendto(data, (target_ip, target_port))
            send_sock.close()
            print(f"[FORWARD] Sent vision data to {target_ip}:{target_port}")
    await loop.create_datagram_endpoint(
        lambda: MatlabReceiver(), local_addr=('0.0.0.0', 8889))

    # 阻塞主线程，保持服务持续运行
    await asyncio.Event().wait()

if __name__ == '__main__':
    asyncio.run(main())
