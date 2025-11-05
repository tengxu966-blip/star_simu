"""
下载指定天区的Gaia星表（示例：下载中心(ra,dec)、视场fov、星等mag限制的星表）
依赖：astroquery、pandas
用法：python download_gaia.py --ra 0 --dec 0 --fov 1 --mag 16 --out gaia.csv
"""
import argparse
from astroquery.gaia import Gaia
import pandas as pd

parser = argparse.ArgumentParser(description='Download Gaia star catalog for a given region.')
parser.add_argument('--ra', type=float, default=0.0, help='中心赤经(度)')
parser.add_argument('--dec', type=float, default=0.0, help='中心赤纬(度)')
parser.add_argument('--fov', type=float, default=1.0, help='视场(度)')
parser.add_argument('--mag', type=float, default=16.0, help='星等上限')
parser.add_argument('--out', type=str, default='gaia.csv', help='输出文件名')
args = parser.parse_args()

# 构造ADQL查询
radius = args.fov / 2
adql = f"""
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {args.ra}, {args.dec}, {radius}))
AND phot_g_mean_mag < {args.mag}
"""
print('[INFO] Querying Gaia...')
job = Gaia.launch_job_async(adql)
df = job.get_results().to_pandas()
print(f'[INFO] Got {len(df)} stars.')
df.rename(columns={'phot_g_mean_mag': 'mag'}, inplace=True)
df.to_csv(args.out, index=False)
print(f'[INFO] Saved to {args.out}')
