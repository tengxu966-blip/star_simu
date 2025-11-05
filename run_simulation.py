"""
一键仿真主控脚本：
- 启动异步UDP服务器
- 随机生成卫星状态并模拟MATLAB端发送
- 接收视觉结果并打印
"""
import asyncio
import random
import struct
from src.protocol import pack_satellite_state, unpack_vision_data
from src.async_server import GNCServerProtocol

async def fake_matlab_sender(loop, addr=('127.0.0.1', 8888)):
    import os
    from src.orbit import TargetOrbit
    sock = asyncio.DatagramProtocol()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: sock, remote_addr=addr)
    own_eph = os.path.join('/Users/zergjh/Documents/sync/今日/star_simu/data', 'Satellite1.e')
    target_eph = os.path.join('/Users/zergjh/Documents/sync/今日/star_simu/data', '2016ho3.e')
    print("own_eph:", own_eph)
    if os.path.exists(own_eph):
        orbit = TargetOrbit(own_eph)
        target_orbit = TargetOrbit(target_eph) if os.path.exists(target_eph) else None
        print(f"[SIM] Using own orbit: {own_eph}, target orbit: {target_eph}")
        for i in range(1440):
            timestamp = i*60  # 每分钟一个状态
            pos, vel = orbit.get_state(timestamp)
            # 目标星位置
            if target_orbit:
                target_pos, _ = target_orbit.get_state(timestamp)
                from src.attitude_utils import look_at_quaternion
                quat = look_at_quaternion(pos, target_pos)
                print(f"[SIM] Target pos at t={timestamp}: {target_pos}, quat={quat}")
            else:
                quat = (0, 0, 0, 1)
            omega = (0, 0, 0)
            data = pack_satellite_state(timestamp, pos, vel, quat, omega)
            transport.sendto(data)
            await asyncio.sleep(0.01)
    # else:
    #     for i in range(3):
    #         # 随机生成卫星状态
    #         timestamp = i
    #         pos = (random.uniform(900, 1100), random.uniform(-100, 100), random.uniform(9500, 10500))
    #         vel = (0, 0, 0)
    #         quat = (0, 0, 0, 1)
    #         omega = (0, 0, 0)
    #         data = pack_satellite_state(timestamp, pos, vel, quat, omega)
    #         transport.sendto(data)
    #         await asyncio.sleep(0.5)
    transport.close()

async def fake_matlab_receiver(loop, listen_port=8889):
    class Receiver(asyncio.DatagramProtocol):
        def datagram_received(self, data, addr):
            msg = unpack_vision_data(data)
            print(f"[MATLAB] Got vision: t={msg.timestamp}, u={msg.u}, v={msg.v}, conf={msg.confidence:.3f}")
    await loop.create_datagram_endpoint(
        lambda: Receiver(), local_addr=('0.0.0.0', listen_port))

async def main():
    loop = asyncio.get_running_loop()
    # 启动GNC异步服务器
    await loop.create_datagram_endpoint(
        lambda: GNCServerProtocol(), local_addr=('0.0.0.0', 8888))
    # 启动MATLAB模拟收发
    await asyncio.gather(
        fake_matlab_sender(loop),
        fake_matlab_receiver(loop)
    )

if __name__ == '__main__':
    asyncio.run(main())
