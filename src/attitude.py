"""
四元数与天球坐标变换
"""
import numpy as np

def quaternion_to_matrix(q):
    """
    四元数(qx,qy,qz,qw)转旋转矩阵
    """
    qx, qy, qz, qw = q
    R = np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]
    ])
    return R

def get_fov_center_ra_dec(pos, quat):
    """
    pos: 卫星位置(可选)
    quat: 卫星姿态四元数(qx,qy,qz,qw)
    返回当前视场中心(ra, dec)（单位度，J2000）
    """
    # 卫星本体坐标系z轴(0,0,1)为视场中心
    R = quaternion_to_matrix(quat)
    z_body = np.array([0,0,1])
    z_inertial = R @ z_body
    # 单位矢量转ra,dec
    x, y, z = z_inertial
    ra = np.degrees(np.arctan2(y, x)) % 360
    dec = np.degrees(np.arcsin(z / np.linalg.norm(z_inertial)))
    return ra, dec

def radec_to_unitvec(ra, dec):
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.array([x, y, z])
