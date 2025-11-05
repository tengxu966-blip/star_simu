import numpy as np

def look_at_quaternion(src_pos, target_pos):
    """
    计算使卫星z轴（机体[0,0,1]）指向目标的四元数(qx,qy,qz,qw)
    src_pos/target_pos: 3元组，单位一致
    返回: (qx, qy, qz, qw)
    """
    z_body = np.array([0,0,1])
    v = np.array(target_pos) - np.array(src_pos)
    v = v / np.linalg.norm(v)
    # 计算旋转轴和角度
    axis = np.cross(z_body, v)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # 已经对准或反向
        if np.dot(z_body, v) > 0:
            return (0,0,0,1)
        else:
            # 180度旋转
            return (1,0,0,0)
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(z_body, v), -1, 1))
    s = np.sin(angle/2)
    qx = axis[0]*s
    qy = axis[1]*s
    qz = axis[2]*s
    qw = np.cos(angle/2)
    return (qx, qy, qz, qw)
