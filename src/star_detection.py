"""
星点检测和可视化工具
从FITS图像序列中提取所有星点并进行可视化
基于SSIM的运动目标检测
"""

import numpy as np
import cv2
import os
import glob
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import label, find_objects
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
import pandas as pd

# 设置matplotlib支持中文
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def load_fits_images(fits_dir):
    """加载FITS图像序列"""
    fits_files = sorted(glob.glob(os.path.join(fits_dir, '*.fits')))
    images = []
    filenames = []
    
    print(f"找到 {len(fits_files)} 个FITS文件")
    
    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                data = hdul[0].data
                if data is None:
                    continue
                
                # 转换为float32并处理NaN
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data = data.astype(np.float32)
                
                images.append(data)
                filenames.append(os.path.basename(fits_file))
                print(f"  加载: {os.path.basename(fits_file)}, 形状: {data.shape}")
        except Exception as e:
            print(f"  错误: 无法加载 {fits_file}: {e}")
            continue
    
    return images, filenames


def normalize_image(image):
    """图像归一化"""
    image = np.array(image, dtype=np.float32)
    if image.max() == image.min():
        return np.zeros_like(image)
    
    # 使用百分位数归一化，避免极值影响
    p2, p98 = np.percentile(image, [2, 98])
    if p98 > p2:
        normalized = (image - p2) / (p98 - p2)
        normalized = np.clip(normalized, 0, 1)
    else:
        normalized = np.zeros_like(image)
    
    return normalized


def calculate_ssim(img1, img2):
    """
    计算两幅图像的结构相似性指数（SSIM）
    
    Parameters:
    -----------
    img1 : np.ndarray
        第一幅图像
    img2 : np.ndarray
        第二幅图像
    
    Returns:
    --------
    ssim : float
        SSIM值，范围在-1到1之间，值越大表示越相似
    """
    # 确保图像大小一致
    if img1.shape != img2.shape:
        # 如果大小不一致，调整较小图像的大小
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    # 转换为float64以避免溢出
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算均值
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # 计算方差和协方差
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM常数（避免分母为0）
    c1 = (0.01 * 255) ** 2  # 对于8位图像
    c2 = (0.03 * 255) ** 2
    
    # 如果图像值范围不是0-255，调整常数
    if img1.max() <= 1.0:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
    
    # 计算SSIM
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim = numerator / (denominator + 1e-10)  # 避免除零
    
    return ssim


def detect_stars(image, threshold_factor=2.0, min_area=1, max_area=1000):
    """
    检测图像中的所有星点，并提取每个星点的小窗口
    
    Parameters:
    -----------
    image : np.ndarray
        输入图像
    threshold_factor : float
        阈值因子（相对于背景标准差）
    min_area : int
        星点最小面积（像素）
    max_area : int
        星点最大面积（像素）
    
    Returns:
    --------
    stars : List[dict]
        检测到的星点列表，每个星点包含：
        - 'position': (x, y) 质心位置
        - 'area': 面积
        - 'intensity': 总强度
        - 'max_intensity': 最大强度
        - 'mean_intensity': 平均强度
        - 'bbox': (x_min, y_min, x_max, y_max) 边界框
        - 'center': (x_center, y_center) 边界框中心
        - 'window_size': (w, h) 窗口大小
        - 'window': np.ndarray 提取的窗口图像
    """
    # 先检查图像的统计信息
    img_min = np.min(image)
    img_max = np.max(image)
    img_mean = np.mean(image)
    img_median = np.median(image)
    img_std = np.std(image)
    
    # 如果图像值很小，可能需要先缩放
    if img_max < 100:
        # 缩放图像以便更好地检测
        image = image * (1000.0 / (img_max + 1e-10))
    
    # 图像归一化
    img_norm = normalize_image(image)
    
    # 计算背景统计量（使用sigma clipping去除星点影响）
    mean, median, std = sigma_clipped_stats(img_norm, sigma=3.0)
    
    # 动态阈值：背景中值 + 阈值因子 * 标准差
    # 对于暗弱目标，降低阈值
    threshold = median + threshold_factor * std
    
    # 如果阈值仍然太高，尝试使用百分位数
    if threshold > 0.9:
        threshold = np.percentile(img_norm, 95)
    
    # 二值化
    binary = (img_norm > threshold).astype(np.uint8)
    
    # 形态学操作：去除小噪声（使用更小的核）
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 连通域标记
    labeled, num_labels = label(binary)
    
    # 提取星点信息
    stars = []
    h, w = image.shape
    
    for i in range(1, num_labels + 1):
        mask = (labeled == i)
        area = np.sum(mask)
        
        # 面积过滤（放宽限制）
        if area < min_area or area > max_area:
            continue
        
        # 计算质心
        y_coords, x_coords = np.where(mask)
        x_centroid = np.mean(x_coords)
        y_centroid = np.mean(y_coords)
        
        # 计算边界框（bounding box）
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        
        # 计算边界框中心
        x_center = x_min + bbox_width / 2
        y_center = y_min + bbox_height / 2
        
        # 窗口大小：(w+4) * (h+4)
        window_width = bbox_width + 4
        window_height = bbox_height + 4
        
        # 计算窗口的边界（以边界框中心为中心）
        window_x_min = max(0, int(x_center - window_width / 2))
        window_y_min = max(0, int(y_center - window_height / 2))
        window_x_max = min(w, int(x_center + window_width / 2))
        window_y_max = min(h, int(y_center + window_height / 2))
        
        # 提取窗口图像
        window = image[window_y_min:window_y_max, window_x_min:window_x_max]
        
        # 计算强度统计（使用原始归一化图像）
        intensity = np.sum(img_norm[mask])
        max_intensity = np.max(img_norm[mask])
        mean_intensity = np.mean(img_norm[mask])
        
        stars.append({
            'position': (x_centroid, y_centroid),
            'area': area,
            'intensity': intensity,
            'max_intensity': max_intensity,
            'mean_intensity': mean_intensity,
            'bbox': (x_min, y_min, x_max, y_max),
            'center': (x_center, y_center),
            'window_size': (window_width, window_height),
            'window': window,
            'window_bbox': (window_x_min, window_y_min, window_x_max, window_y_max),
            'bbox_size': (bbox_width, bbox_height)
        })
    
    return stars, binary, labeled


def detect_moving_targets_by_ssim(stars_by_frame, images, filenames):
    """
    基于SSIM检测运动目标
    逐个比对包含各个星点的窗口之间的结构相似性指数，
    计算每个窗口截取的当前帧图像与前一帧图像这个窗口之间的结构相似性指数，
    记录结构相似性指数最小的窗口，初步认为其是一个运动的目标
    
    Parameters:
    -----------
    stars_by_frame : List[List[dict]]
        每帧检测到的星点列表（包含窗口信息）
    images : List[np.ndarray]
        原始图像序列
    filenames : List[str]
        文件名列表
    
    Returns:
    --------
    moving_targets : List[dict]
        检测到的运动目标列表，每个目标包含：
        - 'frame_idx': 帧索引
        - 'star_idx': 星点索引
        - 'star': 星点信息
        - 'ssim': SSIM值
        - 'position': 位置
    """
    moving_targets = []
    
    for frame_idx in range(1, len(stars_by_frame)):  # 从第二帧开始
        prev_stars = stars_by_frame[frame_idx - 1]
        curr_stars = stars_by_frame[frame_idx]
        prev_image = images[frame_idx - 1]
        curr_image = images[frame_idx]
        
        # 为当前帧的每个星点窗口，计算与前一帧相同位置窗口的SSIM
        ssim_values = []
        
        for curr_star_idx, curr_star in enumerate(curr_stars):
            curr_window = curr_star['window']
            curr_window_bbox = curr_star['window_bbox']
            
            # 在前一帧图像的相同位置提取窗口
            window_x_min, window_y_min, window_x_max, window_y_max = curr_window_bbox
            
            # 提取前一帧相同位置的窗口
            h, w = prev_image.shape
            prev_window_x_min = max(0, window_x_min)
            prev_window_y_min = max(0, window_y_min)
            prev_window_x_max = min(w, window_x_max)
            prev_window_y_max = min(h, window_y_max)
            
            # 如果窗口超出图像边界，跳过
            if prev_window_x_max <= prev_window_x_min or prev_window_y_max <= prev_window_y_min:
                continue
            
            prev_window = prev_image[prev_window_y_min:prev_window_y_max, 
                                    prev_window_x_min:prev_window_x_max]
            
            if prev_window.size == 0 or curr_window.size == 0:
                continue
            
            # 计算SSIM
            ssim = calculate_ssim(curr_window, prev_window)
            
            # 记录SSIM值
            ssim_values.append({
                'frame_idx': frame_idx,
                'star_idx': curr_star_idx,
                'star': curr_star,
                'ssim': ssim,
                'position': curr_star['position']
            })
        
        # 找到SSIM值小于0.4的目标（运动目标）
        if ssim_values:
            for target_info in ssim_values:
                if target_info['ssim'] < 0.4:  # SSIM阈值，小于0.4认为是运动目标
                    moving_targets.append(target_info)
    
    return moving_targets


def track_moving_targets(moving_targets, max_distance=50.0):
    """
    跟踪运动目标，找出轨迹最连续、最平滑的目标
    
    Parameters:
    -----------
    moving_targets : List[dict]
        所有帧中检测到的运动目标列表
    max_distance : float
        相邻帧之间目标的最大距离（像素）
    
    Returns:
    --------
    trajectories : List[dict]
        轨迹列表，每个轨迹包含：
        - 'target_id': 轨迹ID
        - 'frames': 帧索引列表
        - 'positions': 位置列表
        - 'ssim_values': SSIM值列表
        - 'smoothness': 平滑度分数
        - 'continuity': 连续性分数
        - 'avg_speed': 平均速度
    """
    if len(moving_targets) == 0:
        return []
    
    # 按帧分组
    targets_by_frame = {}
    for target in moving_targets:
        frame_idx = target['frame_idx']
        if frame_idx not in targets_by_frame:
            targets_by_frame[frame_idx] = []
        targets_by_frame[frame_idx].append(target)
    
    # 轨迹跟踪
    trajectories = []
    trajectory_id = 0
    
    # 从第一帧开始跟踪
    for start_frame in sorted(targets_by_frame.keys()):
        for start_target in targets_by_frame[start_frame]:
            # 检查是否已经在某个轨迹中
            already_tracked = False
            for traj in trajectories:
                if start_frame in traj['frames']:
                    for idx, f in enumerate(traj['frames']):
                        if f == start_frame:
                            if traj['positions'][idx] == start_target['position']:
                                already_tracked = True
                                break
                    if already_tracked:
                        break
            
            if already_tracked:
                continue
            
            # 开始新轨迹
            trajectory = {
                'target_id': trajectory_id,
                'frames': [start_frame],
                'positions': [start_target['position']],
                'ssim_values': [start_target['ssim']],
                'targets': [start_target]
            }
            trajectory_id += 1
            
            # 在当前帧之后寻找后续目标
            current_pos = start_target['position']
            current_frame = start_frame
            
            while True:
                next_frame = current_frame + 1
                if next_frame not in targets_by_frame:
                    break
                
                # 寻找下一帧中最近的目标
                best_target = None
                best_distance = max_distance + 1
                
                for next_target in targets_by_frame[next_frame]:
                    # 检查是否已被其他轨迹使用
                    used = False
                    for traj in trajectories:
                        if next_frame in traj['frames']:
                            for idx, f in enumerate(traj['frames']):
                                if f == next_frame:
                                    if traj['positions'][idx] == next_target['position']:
                                        used = True
                                        break
                            if used:
                                break
                    
                    if used:
                        continue
                    
                    # 计算距离
                    next_pos = next_target['position']
                    distance = np.sqrt((next_pos[0] - current_pos[0])**2 + 
                                     (next_pos[1] - current_pos[1])**2)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_target = next_target
                
                # 如果找到匹配的目标，添加到轨迹
                if best_target is not None:
                    trajectory['frames'].append(next_frame)
                    trajectory['positions'].append(best_target['position'])
                    trajectory['ssim_values'].append(best_target['ssim'])
                    trajectory['targets'].append(best_target)
                    
                    current_pos = best_target['position']
                    current_frame = next_frame
                else:
                    break
            
            # 只保留长度大于等于2的轨迹
            if len(trajectory['frames']) >= 2:
                trajectories.append(trajectory)
    
    # 计算轨迹的平滑度和连续性
    for trajectory in trajectories:
        positions = np.array(trajectory['positions'])
        frames = np.array(trajectory['frames'])
        
        if len(positions) >= 2:
            # 计算速度
            velocities = []
            for i in range(len(positions) - 1):
                dx = positions[i+1][0] - positions[i][0]
                dy = positions[i+1][1] - positions[i][1]
                dt = frames[i+1] - frames[i]
                if dt > 0:
                    vx = dx / dt
                    vy = dy / dt
                    velocities.append([vx, vy])
            
            if len(velocities) > 0:
                velocities = np.array(velocities)
                
                # 计算平均速度
                avg_velocity = np.mean(velocities, axis=0)
                avg_speed = np.linalg.norm(avg_velocity)
                trajectory['avg_speed'] = avg_speed
                
                # 计算速度一致性（平滑度）
                if len(velocities) > 1:
                    velocity_norms = np.linalg.norm(velocities, axis=1)
                    velocity_dirs = velocities / (velocity_norms[:, None] + 1e-10)
                    
                    # 方向一致性（相邻速度方向的内积）
                    dir_consistency = np.mean([np.dot(velocity_dirs[i], velocity_dirs[i+1]) 
                                             for i in range(len(velocity_dirs)-1)])
                    
                    # 速度稳定性（速度变化系数）
                    speed_std = np.std(velocity_norms)
                    speed_mean = np.mean(velocity_norms)
                    speed_coefficient = speed_std / (speed_mean + 1e-10) if speed_mean > 0 else 1.0
                    
                    # 平滑度分数 = 方向一致性 * (1 - 速度变化系数)
                    smoothness = dir_consistency * (1 - min(speed_coefficient, 1.0))
                else:
                    smoothness = 1.0
                
                trajectory['smoothness'] = smoothness
            else:
                trajectory['avg_speed'] = 0
                trajectory['smoothness'] = 0
            
            # 连续性分数 = 轨迹长度 / 最大可能的轨迹长度
            frame_span = frames[-1] - frames[0] + 1
            continuity = len(frames) / frame_span if frame_span > 0 else 0
            trajectory['continuity'] = continuity
            
            # 综合分数 = 平滑度 * 连续性 * 轨迹长度
            trajectory['score'] = trajectory['smoothness'] * trajectory['continuity'] * len(frames)
        else:
            trajectory['avg_speed'] = 0
            trajectory['smoothness'] = 0
            trajectory['continuity'] = 0
            trajectory['score'] = 0
    
    return trajectories


def visualize_trajectories(images, filenames, stars_by_frame, trajectories, 
                           output_file='trajectories_visualization.png'):
    """
    可视化轨迹
    
    Parameters:
    -----------
    images : List[np.ndarray]
        图像序列
    filenames : List[str]
        文件名列表
    stars_by_frame : List[List[dict]]
        每帧的星点列表
    trajectories : List[dict]
        轨迹列表
    output_file : str
        输出文件名
    """
    num_frames = len(images)
    
    # 计算子图布局（每行最多4个）
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 准备图像用于显示
    def prepare_image(img):
        img = np.array(img)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            p2, p98 = np.percentile(img, [2, 98])
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = (img * 255).astype(np.uint8)
        return img
    
    # 为每个轨迹分配颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # 绘制每一帧
    for frame_idx in range(num_frames):
        row = frame_idx // cols
        col = frame_idx % cols
        
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 显示图像
        img = prepare_image(images[frame_idx])
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        
        # 绘制所有星点（小的点）
        stars = stars_by_frame[frame_idx]
        for star in stars:
            x, y = star['position']
            ax.plot(x, y, 'b.', markersize=1, alpha=0.2)
        
        # 绘制轨迹
        for traj_idx, trajectory in enumerate(trajectories):
            color = colors[traj_idx]
            
            # 绘制到当前帧的轨迹
            positions_up_to_frame = []
            for i, f in enumerate(trajectory['frames']):
                if f <= frame_idx:
                    positions_up_to_frame.append(trajectory['positions'][i])
            
            if len(positions_up_to_frame) > 1:
                pos_array = np.array(positions_up_to_frame)
                ax.plot(pos_array[:, 0], pos_array[:, 1], '-', 
                       color=color, linewidth=2, alpha=0.7, 
                       label=f'轨迹{trajectory["target_id"]+1}')
            
            # 标记当前帧的位置
            if frame_idx in trajectory['frames']:
                idx = trajectory['frames'].index(frame_idx)
                pos = trajectory['positions'][idx]
                ax.plot(pos[0], pos[1], 'o', color=color, markersize=8, 
                       markeredgecolor='white', markeredgewidth=1.5, alpha=0.9)
        
        ax.set_title(f'{filenames[frame_idx]}\n帧 {frame_idx+1}', fontsize=10)
        ax.axis('off')
    
    # 添加图例
    if len(trajectories) > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=min(len(trajectories), 10), 
                     fontsize=8, bbox_to_anchor=(0.5, 0.98))
    
    # 隐藏多余的子图
    for idx in range(num_frames, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n轨迹可视化已保存到: {output_file}")
    plt.close()


def visualize_moving_targets(images, filenames, stars_by_frame, moving_targets, 
                            output_file='moving_targets_detection.png'):
    """
    可视化检测到的运动目标
    
    Parameters:
    -----------
    images : List[np.ndarray]
        图像序列
    filenames : List[str]
        文件名列表
    stars_by_frame : List[List[dict]]
        每帧的星点列表
    moving_targets : List[dict]
        检测到的运动目标列表
    output_file : str
        输出文件名
    """
    num_frames = len(images)
    
    # 计算子图布局（每行最多4个）
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 准备图像用于显示
    def prepare_image(img):
        img = np.array(img)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            p2, p98 = np.percentile(img, [2, 98])
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = (img * 255).astype(np.uint8)
        return img
    
    # 按帧分组运动目标
    targets_by_frame = {}
    for target in moving_targets:
        frame_idx = target['frame_idx']
        if frame_idx not in targets_by_frame:
            targets_by_frame[frame_idx] = []
        targets_by_frame[frame_idx].append(target)
    
    # 绘制每一帧
    for frame_idx in range(num_frames):
        row = frame_idx // cols
        col = frame_idx % cols
        
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 显示图像
        img = prepare_image(images[frame_idx])
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        
        # 绘制所有星点（小的点）
        stars = stars_by_frame[frame_idx]
        for star in stars:
            x, y = star['position']
            ax.plot(x, y, 'b.', markersize=2, alpha=0.3)
        
        # 标记运动目标（大的红色标记）
        if frame_idx in targets_by_frame:
            targets = targets_by_frame[frame_idx]
            for target in targets:
                x, y = target['position']
                star = target['star']
                
                # 绘制窗口边界框
                window_bbox = star['window_bbox']
                rect = Rectangle((window_bbox[0], window_bbox[1]), 
                               window_bbox[2] - window_bbox[0],
                               window_bbox[3] - window_bbox[1],
                               fill=False, edgecolor='red', linewidth=2, linestyle='-')
                ax.add_patch(rect)
                
                # 标记中心点
                ax.plot(x, y, 'r*', markersize=15, markeredgecolor='yellow', 
                       markeredgewidth=1.5, alpha=0.9)
                
                # 添加SSIM值标注
                ax.text(x + 10, y - 10, f"SSIM: {target['ssim']:.3f}", 
                       color='yellow', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        ax.set_title(f'{filenames[frame_idx]}\n检测到 {len(stars)} 个星点' + 
                    (f'\n{len(targets_by_frame.get(frame_idx, []))} 个运动目标' 
                     if frame_idx in targets_by_frame else ''), 
                    fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_frames, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n运动目标检测结果已保存到: {output_file}")
    plt.close()


def visualize_stars(images, filenames, stars_by_frame, output_file='star_detection_results.png'):
    num_frames = len(images)
    
    # 计算子图布局（每行最多4个）
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # 如果只有一行，确保axes是二维数组
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 准备图像用于显示
    def prepare_image(img):
        img = np.array(img)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            p2, p98 = np.percentile(img, [2, 98])
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = (img * 255).astype(np.uint8)
        return img
    
    # 绘制每一帧
    for frame_idx in range(num_frames):
        row = frame_idx // cols
        col = frame_idx % cols
        
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 显示图像
        img = prepare_image(images[frame_idx])
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        
        # 绘制星点
        stars = stars_by_frame[frame_idx]
        for star in stars:
            x, y = star['position']
            # 根据强度设置点的大小
            intensity_norm = star['max_intensity']
            marker_size = 5 + intensity_norm * 20
            
            ax.plot(x, y, 'ro', markersize=marker_size, 
                   markeredgecolor='yellow', markeredgewidth=0.5, alpha=0.8)
        
        ax.set_title(f'{filenames[frame_idx]}\n检测到 {len(stars)} 个星点', 
                    fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_frames, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_file}")
    plt.close()


def visualize_stars_statistics(stars_by_frame, filenames, output_file='star_statistics.png'):
    """可视化星点统计信息"""
    num_frames = len(stars_by_frame)
    
    # 统计每帧的星点数量
    star_counts = [len(stars) for stars in stars_by_frame]
    
    # 统计每帧的星点总强度
    total_intensities = [np.sum([s['intensity'] for s in stars]) 
                        for stars in stars_by_frame]
    
    # 统计每帧的平均星点强度
    avg_intensities = [np.mean([s['mean_intensity'] for s in stars]) 
                      if len(stars) > 0 else 0
                      for stars in stars_by_frame]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 星点数量统计
    axes[0].plot(range(num_frames), star_counts, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('帧编号', fontsize=12)
    axes[0].set_ylabel('星点数量', fontsize=12)
    axes[0].set_title('每帧检测到的星点数量', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(num_frames))
    
    # 总强度统计
    axes[1].plot(range(num_frames), total_intensities, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('帧编号', fontsize=12)
    axes[1].set_ylabel('总强度', fontsize=12)
    axes[1].set_title('每帧星点总强度', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(num_frames))
    
    # 平均强度统计
    axes[2].plot(range(num_frames), avg_intensities, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('帧编号', fontsize=12)
    axes[2].set_ylabel('平均强度', fontsize=12)
    axes[2].set_title('每帧星点平均强度', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(num_frames))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"统计图表已保存到: {output_file}")
    plt.close()


def plot_trajectories(images, filenames, best_trajectories, output_file='trajectories_plot.png'):
    """
    绘制星点轨迹图
    
    Parameters:
    -----------
    images : List[np.ndarray]
        图像序列
    filenames : List[str]
        文件名列表
    best_trajectories : List[dict]
        最连续、最平滑的3个目标轨迹
    output_file : str
        输出文件名
    """
    if len(best_trajectories) == 0:
        print("没有轨迹数据，跳过轨迹图生成")
        return
    
    # 创建一个大的图像，显示所有轨迹
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 使用第一帧图像作为背景
    background_image = images[0]
    
    # 准备图像用于显示
    def prepare_image(img):
        img = np.array(img)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            p2, p98 = np.percentile(img, [2, 98])
            img = np.clip((img - p2) / (p98 - p2 + 1e-10), 0, 1)
            img = (img * 255).astype(np.uint8)
        return img
    
    bg_img = prepare_image(background_image)
    ax.imshow(bg_img, cmap='gray', vmin=0, vmax=255, alpha=0.5)
    
    # 为每个轨迹分配颜色
    colors = ['red', 'lime', 'cyan']
    markers = ['o', 's', '^']
    
    # 绘制每个轨迹
    for traj_idx, trajectory in enumerate(best_trajectories):
        color = colors[traj_idx % len(colors)]
        marker = markers[traj_idx % len(markers)]
        
        positions = np.array(trajectory['positions'])
        frames = trajectory['frames']
        
        # 绘制轨迹线
        ax.plot(positions[:, 0], positions[:, 1], '-', 
               color=color, linewidth=2.5, alpha=0.8, 
               label=f'轨迹{traj_idx+1} (长度:{len(frames)}帧)')
        
        # 绘制起点
        ax.plot(positions[0, 0], positions[0, 1], marker=marker, 
               color=color, markersize=12, markeredgecolor='white', 
               markeredgewidth=2, label=f'轨迹{traj_idx+1}起点', zorder=5)
        
        # 绘制终点
        ax.plot(positions[-1, 0], positions[-1, 1], marker=marker, 
               color=color, markersize=12, markeredgecolor='black', 
               markeredgewidth=2, label=f'轨迹{traj_idx+1}终点', zorder=5)
        
        # 绘制中间点
        if len(positions) > 2:
            ax.plot(positions[1:-1, 0], positions[1:-1, 1], 'o', 
                   color=color, markersize=6, alpha=0.6, zorder=4)
        
        # 添加帧号标注（每隔几帧标注一次）
        for i in range(0, len(positions), max(1, len(positions) // 5)):
            ax.text(positions[i, 0] + 5, positions[i, 1] - 5, 
                   f"F{frames[i]}", color=color, fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('X (像素)', fontsize=12)
    ax.set_ylabel('Y (像素)', fontsize=12)
    ax.set_title('星点轨迹图', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # 图像坐标系Y轴向下
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"轨迹图已保存到: {output_file}")
    plt.close()


def export_to_excel(stars_by_frame, filenames, best_trajectories, output_file='detection_results.xlsx'):
    """
    导出所有星点和轨迹信息到Excel文件
    
    Parameters:
    -----------
    stars_by_frame : List[List[dict]]
        每帧的星点列表
    filenames : List[str]
        文件名列表
    best_trajectories : List[dict]
        最连续、最平滑的3个目标轨迹
    output_file : str
        输出Excel文件名
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 1. 所有星点信息
        stars_data = []
        for frame_idx, (stars, filename) in enumerate(zip(stars_by_frame, filenames)):
            for star in stars:
                stars_data.append({
                    'frame_number': frame_idx,
                    'filename': filename,
                    'x_pixel': star['position'][0],
                    'y_pixel': star['position'][1],
                    'area': star['area'],
                    'intensity': star['intensity'],
                    'max_intensity': star['max_intensity'],
                    'mean_intensity': star['mean_intensity'],
                    'bbox_x_min': star['bbox'][0],
                    'bbox_y_min': star['bbox'][1],
                    'bbox_x_max': star['bbox'][2],
                    'bbox_y_max': star['bbox'][3],
                    'window_width': star['window_size'][0],
                    'window_height': star['window_size'][1]
                })
        
        stars_df = pd.DataFrame(stars_data)
        stars_df.to_excel(writer, sheet_name='所有星点', index=False)
        
        # 2. 轨迹信息
        if len(best_trajectories) > 0:
            trajectories_data = []
            for traj_idx, traj in enumerate(best_trajectories, 1):
                for i, (frame, pos, ssim) in enumerate(zip(traj['frames'], traj['positions'], traj['ssim_values'])):
                    trajectories_data.append({
                        'trajectory_id': traj_idx,
                        'frame_number': frame,
                        'x_pixel': pos[0],
                        'y_pixel': pos[1],
                        'ssim': ssim,
                        'trajectory_length': len(traj['frames']),
                        'smoothness': traj['smoothness'],
                        'continuity': traj['continuity'],
                        'score': traj['score'],
                        'avg_speed': traj['avg_speed'],
                        'avg_ssim': np.mean(traj['ssim_values'])
                    })
            
            trajectories_df = pd.DataFrame(trajectories_data)
            trajectories_df.to_excel(writer, sheet_name='轨迹', index=False)
            
            # 3. 轨迹摘要
            summary_data = []
            for traj_idx, traj in enumerate(best_trajectories, 1):
                summary_data.append({
                    'trajectory_id': traj_idx,
                    'trajectory_length': len(traj['frames']),
                    'frame_start': traj['frames'][0],
                    'frame_end': traj['frames'][-1],
                    'smoothness': traj['smoothness'],
                    'continuity': traj['continuity'],
                    'score': traj['score'],
                    'avg_speed': traj['avg_speed'],
                    'avg_ssim': np.mean(traj['ssim_values']),
                    'start_x': traj['positions'][0][0],
                    'start_y': traj['positions'][0][1],
                    'end_x': traj['positions'][-1][0],
                    'end_y': traj['positions'][-1][1]
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='轨迹摘要', index=False)
        else:
            # 如果没有轨迹，创建空表
            empty_df = pd.DataFrame(columns=['trajectory_id', 'frame_number', 'x_pixel', 'y_pixel', 'ssim'])
            empty_df.to_excel(writer, sheet_name='轨迹', index=False)
            
            empty_summary_df = pd.DataFrame(columns=['trajectory_id', 'trajectory_length', 'smoothness', 'continuity'])
            empty_summary_df.to_excel(writer, sheet_name='轨迹摘要', index=False)
    
    print(f"结果已导出到: {output_file}")


def main():
    """主函数"""
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 1. 加载FITS图像（相对于项目根目录）
    fits_dir = os.path.join(project_root, 'fits_images')
    images, filenames = load_fits_images(fits_dir)
    
    if len(images) == 0:
        print("错误: 未找到FITS图像文件")
        return
    
    # 2. 检测每帧的星点
    stars_by_frame = []
    
    for frame_idx, (image, filename) in enumerate(zip(images, filenames)):
        # 检测星点
        stars, binary, labeled = detect_stars(image, 
                                              threshold_factor=1.5,
                                              min_area=1,
                                              max_area=1000)
        
        stars_by_frame.append(stars)
    
    # 3. 基于SSIM检测运动目标
    moving_targets = detect_moving_targets_by_ssim(stars_by_frame, images, filenames)
    
    # 4. 轨迹跟踪，找出最连续、最平滑的3个目标
    all_trajectories = track_moving_targets(moving_targets, max_distance=50.0)
    
    # 按综合分数排序，找出最好的3个轨迹
    sorted_trajectories = sorted(all_trajectories, key=lambda x: x['score'], reverse=True)
    best_trajectories = sorted_trajectories[:3]
    
    # 5. 绘制轨迹图（输出到项目根目录）
    output_dir = project_root
    plot_trajectories(images, filenames, best_trajectories, 
                     output_file=os.path.join(output_dir, 'trajectories_plot.png'))
    
    # 6. 导出到Excel（输出到项目根目录）
    export_to_excel(stars_by_frame, filenames, best_trajectories, 
                   output_file=os.path.join(output_dir, 'detection_results.xlsx'))
    
    print("\n处理完成！")
    excel_path = os.path.join(output_dir, 'detection_results.xlsx')
    plot_path = os.path.join(output_dir, 'trajectories_plot.png')
    print(f"Excel文件已生成: {excel_path}")
    print(f"  - 工作表1: 所有星点")
    print(f"  - 工作表2: 轨迹")
    print(f"  - 工作表3: 轨迹摘要")
    print(f"轨迹图已生成: {plot_path}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        input("按Enter键退出...")

