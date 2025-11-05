"""
目标检测算法 - 基于star_detection.py的星点检测和运动目标检测
"""

import numpy as np
import cv2
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import label


# 全局变量：用于多帧SSIM检测的帧缓存
_frame_cache = []
_max_cache_size = 5


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
    
    return stars


def detect_moving_target_single_frame(image, prev_image=None, threshold_factor=2.0, min_area=1, max_area=1000):
    """
    单帧或双帧模式下的运动目标检测
    
    Parameters:
    -----------
    image : np.ndarray
        当前帧图像
    prev_image : np.ndarray, optional
        前一帧图像（如果提供，将使用SSIM检测运动目标）
    threshold_factor : float
        阈值因子
    min_area : int
        星点最小面积
    max_area : int
        星点最大面积
    
    Returns:
    --------
    target_x : int
        目标像素x坐标
    target_y : int
        目标像素y坐标
    confidence : float
        置信度（0-1）
    """
    # 检测当前帧的所有星点
    stars = detect_stars(image, threshold_factor=threshold_factor, min_area=min_area, max_area=max_area)
    
    if len(stars) == 0:
        # 如果没有检测到星点，返回图像中心
        h, w = image.shape
        return int(w // 2), int(h // 2), 0.0
    
    # 如果有前一帧，使用SSIM检测运动目标
    if prev_image is not None:
        moving_candidates = []
        
        for star in stars:
            curr_window = star['window']
            window_bbox = star['window_bbox']
            window_x_min, window_y_min, window_x_max, window_y_max = window_bbox
            
            # 在前一帧图像的相同位置提取窗口
            h, w = prev_image.shape
            prev_window_x_min = max(0, window_x_min)
            prev_window_y_min = max(0, window_y_min)
            prev_window_x_max = min(w, window_x_max)
            prev_window_y_max = min(h, window_y_max)
            
            if prev_window_x_max <= prev_window_x_min or prev_window_y_max <= prev_window_y_min:
                continue
            
            prev_window = prev_image[prev_window_y_min:prev_window_y_max, 
                                    prev_window_x_min:prev_window_x_max]
            
            if prev_window.size == 0 or curr_window.size == 0:
                continue
            
            # 计算SSIM
            ssim = calculate_ssim(curr_window, prev_window)
            
            # SSIM值越小，说明变化越大，越可能是运动目标
            # 阈值设为0.4，小于此值认为是运动目标
            if ssim < 0.4:
                moving_candidates.append({
                    'star': star,
                    'ssim': ssim,
                    'motion_score': 1.0 - ssim  # 运动分数：1-SSIM，值越大越可能是运动目标
                })
        
        # 如果检测到运动目标，选择运动分数最高的
        if len(moving_candidates) > 0:
            best_candidate = max(moving_candidates, key=lambda x: x['motion_score'])
            best_star = best_candidate['star']
            x, y = best_star['position']
            confidence = min(1.0, best_candidate['motion_score'] * 2.0)  # 将运动分数转换为置信度
            return int(round(x)), int(round(y)), confidence
    
    # 如果没有前一帧或没有检测到运动目标，选择最亮的星点作为目标
    # 根据最大强度和总强度的综合评分
    best_star = max(stars, key=lambda s: s['max_intensity'] * 0.7 + s['intensity'] * 0.3)
    x, y = best_star['position']
    
    # 置信度基于星点的亮度（归一化）
    max_possible_intensity = max([s['max_intensity'] for s in stars]) if stars else 1.0
    confidence = min(1.0, best_star['max_intensity'] / (max_possible_intensity + 1e-10))
    
    return int(round(x)), int(round(y)), confidence


def detect_asteroid(img):
    """
    检测小行星目标（主接口函数）
    
    Parameters:
    -----------
    img : np.ndarray
        输入图像（numpy数组，可以是uint8或float格式）
    
    Returns:
    --------
    u : int
        目标像素x坐标
    v : int
        目标像素y坐标
    confidence : float
        置信度（0-1）
    """
    global _frame_cache
    
    # 确保图像是numpy数组
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # 转换为float32以便处理
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # 获取前一帧（如果存在）
    prev_image = None
    if len(_frame_cache) > 0:
        prev_image = _frame_cache[-1]
    
    # 检测目标
    u, v, confidence = detect_moving_target_single_frame(
        img, 
        prev_image=prev_image,
        threshold_factor=2.0,
        min_area=1,
        max_area=1000
    )
    
    # 更新帧缓存（保留最近几帧）
    _frame_cache.append(img.copy())
    if len(_frame_cache) > _max_cache_size:
        _frame_cache.pop(0)
    
    return u, v, confidence
