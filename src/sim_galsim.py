"""
小行星图像生成接口（galsim stub）
"""


import galsim
import numpy as np
import toml
import os

# 全局缓存配置和背景星
_sim_config = None
_cached_stars = None
_cached_gaia_path = None
_cached_gaia_df = None

def generate_asteroid_image(sat_msg, gaia_catalog_path=None, fov_deg=0.5):
    from src.gaia_tools import download_gaia_catalog
    from src.attitude import get_fov_center_ra_dec, radec_to_unitvec
    global _sim_config, _cached_stars, _cached_gaia_path, _cached_gaia_df
    # 只读取一次配置
    if _sim_config is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sim_config.toml')
        if os.path.exists(config_path):
            _sim_config = toml.load(config_path)
        else:
            _sim_config = {}
    use_gaia_background = _sim_config.get('use_gaia_background', True)
    config_gaia_path = _sim_config.get('gaia_catalog_path', None)
    config_fov_deg = _sim_config.get('fov_deg', 3.0)
    n_stars = _sim_config.get('n_stars', 50)
    gaia_catalog_path = gaia_catalog_path or config_gaia_path or None
    fov_deg = fov_deg if fov_deg != 0.5 else config_fov_deg
    img_size = 2048
    cx, cy = img_size // 2, img_size // 2
    # 计算像元尺寸（arcsec/pixel）
    fov_deg = float(fov_deg)
    pixel_scale = (fov_deg * 3600) / img_size  # 单位: arcsec/pixel
    """
def generate_asteroid_image(sat_msg, gaia_catalog_path=None, fov_deg=0.5, mag_limit=16):
    2. galsim生成小行星目标图像
    3. 可选：叠加gaia星表背景星
    gaia_catalog_path: 可选，csv或txt，包含ra,dec,mag三列
    fov_deg: 视场角，单位度
    """
    # 1. 相对位置
    rel_x = -sat_msg.pos_x
    rel_y = -sat_msg.pos_y
    rel_z = -sat_msg.pos_z
    # 2. 简单针孔投影到像平面
    f = 100.0
    if rel_z == 0:
        rel_z = 1e-6
    u = f * rel_x / rel_z
    v = f * rel_y / rel_z
    px = int(round(cx + u))
    py = int(round(cy + v))
    # 3. galsim生成星点
    image = galsim.ImageF(img_size, img_size, scale=pixel_scale)
    psf = galsim.Gaussian(fwhm=2*pixel_scale)  # PSF宽度约2像元
    # 小行星
    asteroid = psf.withFlux(10000)
    asteroid.drawImage(image=image, offset=(px-cx, py-cy), add_to_image=True)

    # 4. 叠加gaia星表背景星或自动生成背景星
    if not use_gaia_background:
        arr = np.clip(image.array, 0, 255).astype(np.uint8)
        return arr, None  # 添加None作为vis_img的返回值

    ra0, dec0 = 0.0, 0.0
    fov = fov_deg
    df = None
    # 优先缓存gaia星表
    if gaia_catalog_path:
        if _cached_gaia_path != gaia_catalog_path or _cached_gaia_df is None:
            try:
                import pandas as pd
                _cached_gaia_df = pd.read_csv(gaia_catalog_path)
                _cached_gaia_path = gaia_catalog_path
                # 自动保存一份到data/gaia_catalog_checked.csv
                data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
                os.makedirs(data_dir, exist_ok=True)
                checked_path = os.path.join(data_dir, 'gaia_catalog_checked.csv')
                _cached_gaia_df.to_csv(checked_path, index=False)
                print(f"[INFO] Gaia星表已保存到: {checked_path}")
            except Exception as e:
                print(f"[WARN] Gaia星表加载失败: {e}")
                _cached_gaia_df = None
        df = _cached_gaia_df
    else:
        # 缓存背景星
        if _cached_stars is None:
            rng = np.random.default_rng(42)
            ras = ra0 + (rng.random(n_stars) - 0.5) * fov
            decs = dec0 + (rng.random(n_stars) - 0.5) * fov
            mags = rng.uniform(8, 15, n_stars)
            try:
                import pandas as pd
                _cached_stars = pd.DataFrame({'ra': ras, 'dec': decs, 'mag': mags})
            except ImportError:
                # 兼容无pandas环境
                _cached_stars = [{'ra': ras[i], 'dec': decs[i], 'mag': mags[i]} for i in range(n_stars)]
        df = _cached_stars

    # 叠加背景星
    if df is not None:
        # 兼容pandas和list两种df
        rows = df.iterrows() if hasattr(df, 'iterrows') else enumerate(df)
        for _, row in rows:
            ra = row['ra'] if isinstance(row, dict) else row.ra
            dec = row['dec'] if isinstance(row, dict) else row.dec
            mag = row['mag'] if isinstance(row, dict) else row.mag
            dra = (ra - ra0) * np.cos(np.deg2rad(dec0))
            ddec = dec - dec0
            x = int(cx + (dra / fov) * img_size)
            y = int(cy + (ddec / fov) * img_size)
            if 0 <= x < img_size and 0 <= y < img_size:
                flux = 10**(6 - 0.4*mag)
                star = psf.withFlux(flux)
                star.drawImage(image=image, offset=(x-cx, y-cy), add_to_image=True)

    arr = np.clip(image.array, 0, 255).astype(np.uint8)

    # 生成可视化结果图：圈出目标、标注目标、标注背景星
    from PIL import Image, ImageDraw, ImageFont
    vis_img = Image.fromarray(arr).convert('RGB')
    draw = ImageDraw.Draw(vis_img)
    # 目标像素
    cx, cy = img_size // 2, img_size // 2
    rel_x = -sat_msg.pos_x
    rel_y = -sat_msg.pos_y
    rel_z = -sat_msg.pos_z
    f = 100.0
    if rel_z == 0:
        rel_z = 1e-6
    u = f * rel_x / rel_z
    v = f * rel_y / rel_z
    px = int(round(cx + u))
    py = int(round(cy + v))
    # 红色矩形框标注目标星
    draw.rectangle([(px-8, py-8), (px+8, py+8)], outline=(255,0,0), width=2)
    draw.text((px+12, py-12), 'Target', fill=(255,0,0))
    # 标注背景星（绿色圆圈+唯一id，优先source_id，其次id，再次index）
    if df is not None:
        if hasattr(df, 'iterrows'):
            for idx, row in df.iterrows():
                ra = row['ra']
                dec = row['dec']
                if 'source_id' in row:
                    star_id = row['source_id']
                elif 'id' in row:
                    star_id = row['id']
                else:
                    star_id = idx
                dra = (ra - ra0) * np.cos(np.deg2rad(dec0))
                ddec = dec - dec0
                x = int(cx + (dra / fov) * img_size)
                y = int(cy + (ddec / fov) * img_size)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw.ellipse([(x-4, y-4), (x+4, y+4)], outline=(0,255,0), width=1)
                    draw.text((x+6, y-6), str(star_id), fill=(0,128,0))
        else:
            for idx, row in enumerate(df):
                ra = row['ra']
                dec = row['dec']
                if 'source_id' in row:
                    star_id = row['source_id']
                elif 'id' in row:
                    star_id = row['id']
                else:
                    star_id = idx
                dra = (ra - ra0) * np.cos(np.deg2rad(dec0))
                ddec = dec - dec0
                x = int(cx + (dra / fov) * img_size)
                y = int(cy + (ddec / fov) * img_size)
                if 0 <= x < img_size and 0 <= y < img_size:
                    draw.ellipse([(x-4, y-4), (x+4, y+4)], outline=(0,255,0), width=1)
                    draw.text((x+6, y-6), str(star_id), fill=(0,128,0))

    return arr, vis_img