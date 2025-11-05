import asyncio
from src.protocol import unpack_satellite_state, pack_vision_data
from src.sim_galsim import generate_asteroid_image
from src.detector import detect_asteroid

class GNCServerProtocol(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        self.transport = transport
        print("UDP server started.")

    def datagram_received(self, data, addr):
        try:
            # 1. 解包卫星状态
            print(f"[RECV] from {addr}: raw={data}")
            sat_msg = unpack_satellite_state(data)
            print(f"[UNPACK] SatelliteState: {sat_msg}")
            print(f"Received from {addr}: timestamp={sat_msg.timestamp}")
            # 2. 查询目标卫星轨道状态
            import toml, os
            from src.orbit import TargetOrbit
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sim_config.toml')
            if os.path.exists(config_path):
                config = toml.load(config_path)
                target_eph = config.get('target_orbit_file', None)
            else:
                target_eph = None
            target_pos, target_vel = (None, None)
            if target_eph:
                try:
                    orbit = TargetOrbit(target_eph)
                    target_pos, target_vel = orbit.get_state(getattr(sat_msg, 'timestamp', 0))
                    print(f"[ORBIT] Target at t={getattr(sat_msg, 'timestamp', 0)}: pos={target_pos}, vel={target_vel}")
                except Exception as e:
                    print(f"[ORBIT] Failed to get target state: {e}")
            # 3. 用galsim生成图像（可扩展为用目标状态）
            img, vis_img = generate_asteroid_image(sat_msg)
            # 保存原始图像和可视化结果图
            import os, json
            from PIL import Image
            import math
            res_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'res')
            os.makedirs(res_dir, exist_ok=True)
            img_name = f"asteroid_{sat_msg.timestamp}.png"
            img_path = os.path.join(res_dir, img_name)
            im = Image.fromarray(img)
            im.save(img_path)
            # 保存可视化结果图
            vis_name = f"asteroid_{sat_msg.timestamp}_result.png"
            vis_path = os.path.join(res_dir, vis_name)
            vis_img.save(vis_path)
            # 计算真值目标像素（假设中心）
            cx, cy = 128, 128
            rel_x = -sat_msg.pos_x
            rel_y = -sat_msg.pos_y
            rel_z = -sat_msg.pos_z
            f = 100.0
            if rel_z == 0:
                rel_z = 1e-6
            u_true = f * rel_x / rel_z
            v_true = f * rel_y / rel_z
            px_true = int(round(cx + u_true))
            py_true = int(round(cy + v_true))
            angle_true = math.degrees(math.atan2(v_true, u_true)) if (u_true != 0 or v_true != 0) else 0.0
            vel = getattr(sat_msg, 'vel', None)
            if vel is None and hasattr(sat_msg, 'vel_x'):
                vel = (sat_msg.vel_x, sat_msg.vel_y, sat_msg.vel_z)
            from src.sim_galsim import _cached_stars
            bgstars = []
            if _cached_stars is not None:
                if hasattr(_cached_stars, 'sort_values'):
                    top3 = _cached_stars.sort_values('mag').head(3)
                    for idx, row in top3.iterrows():
                        bgstars.append({
                            'id': int(idx),
                            'ra': float(row['ra']),
                            'dec': float(row['dec']),
                            'mag': float(row['mag'])
                        })
                else:
                    top3 = sorted(_cached_stars, key=lambda x: x['mag'])[:3]
                    for i, row in enumerate(top3):
                        bgstars.append({
                            'id': i,
                            'ra': float(row['ra']),
                            'dec': float(row['dec']),
                            'mag': float(row['mag'])
                        })
            fov = 0.5
            for star in bgstars:
                dra = (star['ra'] - 0.0) * math.cos(math.radians(0.0))
                ddec = star['dec'] - 0.0
                x = int(cx + (dra / fov) * 256)
                y = int(cy + (ddec / fov) * 256)
                star['pixel_x'] = x
                star['pixel_y'] = y
            truth = {
                'timestamp': getattr(sat_msg, 'timestamp', None),
                'vel': list(vel) if vel is not None else None,
                'target_pixel': [px_true, py_true],
                'angle_deg': angle_true,
                'background_stars': [
                    {'id': s['id'], 'pixel_x': s['pixel_x'], 'pixel_y': s['pixel_y']} for s in bgstars
                ],
                'target_orbit_pos': list(target_pos) if target_pos is not None else None,
                'target_orbit_vel': list(target_vel) if target_vel is not None else None
            }
            json_name = f"asteroid_{sat_msg.timestamp}_truth.json"
            json_path = os.path.join(res_dir, json_name)
            with open(json_path, 'w') as f:
                json.dump(truth, f, indent=2)
            print(f"[INFO] 图像和真值已保存: {img_name}, {json_name}")

            # 3. 目标检测
            try:
                u, v, conf = detect_asteroid(img)
                # 保存算法输出
                result = {
                    'timestamp': getattr(sat_msg, 'timestamp', None),
                    'detect_pixel': [u, v],
                    'confidence': conf
                }
                result_name = f"asteroid_{sat_msg.timestamp}_result.json"
                result_path = os.path.join(res_dir, result_name)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"[INFO] 检测结果已保存: {result_name}")
            except Exception as e:
                print(f"[WARN] detect_asteroid failed: {e}")
                return
            # 4. 打包并返回
            resp = pack_vision_data(sat_msg.timestamp, u, v, conf)
            # 读取配置文件中的目标IP和端口
            import toml, socket, os
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sim_config.toml')
            if os.path.exists(config_path):
                config = toml.load(config_path)
                target_ip = config.get('vision_target_ip', '192.168.29.235')
                target_port = config.get('vision_target_port', 8889)
            else:
                target_ip = '192.168.29.235'
                target_port = 8889
            import time
            wait_sec = config.get('send_wait_seconds', 0)
            if wait_sec:
                print(f"[WAIT] 等待 {wait_sec} 秒后发送视觉结果...")
                time.sleep(wait_sec)
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            send_sock.sendto(resp, (target_ip, target_port))
            send_sock.close()
            print(f"[SEND] to {target_ip}:{target_port}: raw={resp}")
            print(f"Sent result: u={u}, v={v}, conf={conf:.3f}")
            # 打印解包后的视觉数据
            from src.protocol import unpack_vision_data
            try:
                msg = unpack_vision_data(resp)
                print(f"[UNPACK] VisionData: {msg}")
            except Exception as e:
                print(f"[UNPACK] VisionData failed: {e}")
        except Exception as e:
            print(f"Error processing packet: {e}")

async def main():
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: GNCServerProtocol(),
        local_addr=('0.0.0.0', 8888)
    )
    try:
        await asyncio.sleep(3600*24)
    finally:
        transport.close()

if __name__ == '__main__':
    asyncio.run(main())
