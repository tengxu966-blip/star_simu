"""
轨道模块：支持读取目标卫星的轨道文件（.e），按时间戳查询目标卫星状态
支持STK/GMAT/SSAPy等常见ephemeris格式（假设为文本，列为epoch(s), x, y, z, vx, vy, vz）
"""
import numpy as np
import pandas as pd

class TargetOrbit:
    def __init__(self, eph_file):
        # 自动查找EphemerisTimePosVel，定位数据区
        with open(eph_file, 'r') as f:
            lines = f.readlines()
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == 'EphemerisTimePosVel':
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError('EphemerisTimePosVel not found in orbit file')
        # 数据区：从data_start开始，直到遇到下一个非数据区标记或文件结尾
        data_lines = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
            if line.strip().startswith('#') or line.strip().startswith('END') or line.strip().startswith('Ephemeris'):  # 终止条件
                break
            # 只保留以数字开头的行
            if line.strip()[0] in '-0123456789':
                data_lines.append(line)
        # 用pandas读取，手动指定列名
        from io import StringIO
        colnames = ['epoch', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        content = ''.join(data_lines)
        self.df = pd.read_csv(StringIO(content), sep=r'\s+', names=colnames, engine='python')
        self.time_col = 'epoch'

    def get_state(self, timestamp):
        # 最近邻插值
        idx = (np.abs(self.df[self.time_col] - timestamp)).idxmin()
        row = self.df.loc[idx]
        pos = (row['x'], row['y'], row['z'])
        vel = (row['vx'], row['vy'], row['vz'])
        return pos, vel

# 用法示例：
# orbit = TargetOrbit('target.e')
# pos, vel = orbit.get_state(12345)
