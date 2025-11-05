"""
Gaia星表在线查询与本地缓存
"""
from astroquery.gaia import Gaia
import pandas as pd
import os

def download_gaia_catalog(ra, dec, fov_deg=1.0, mag_limit=16, cache_path=None):
    """
    查询指定天区的Gaia星表，返回DataFrame，支持本地缓存
    ra, dec: 视场中心（度）
    fov_deg: 视场直径（度）
    mag_limit: 星等上限
    cache_path: 若指定则优先读写本地csv
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    query = f"""
    SELECT ra, dec, phot_g_mean_mag as mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {ra}, {dec}, {fov_deg/2})
    )
    AND phot_g_mean_mag < {mag_limit}
    """
    job = Gaia.launch_job_async(query)
    tbl = job.get_results().to_pandas()
    if cache_path:
        tbl.to_csv(cache_path, index=False)
    return tbl
