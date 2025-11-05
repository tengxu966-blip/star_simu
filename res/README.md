# 数据集说明

本目录为仿真自动生成的数据集，结构与内容如下：

## 目录结构

- `asteroid_<t>.png`：原始模拟图像，t为时间戳/帧号。
- `asteroid_<t>_result.png`：结果可视化图像，红框为目标，绿色圈为背景星，标注为星表唯一id。
- `asteroid_<t>_truth.json`：真值文件，包含目标像素、轨道状态、背景星像素等。
- `asteroid_<t>_result.json`：算法检测结果（如有），包含检测像素、置信度等。
- `dataset_index.json`：数据集索引文件，汇总所有帧的文件名、真值、检测结果等。

## asteroid_<t>_truth.json 字段说明
- `timestamp`：帧号/时间戳
- `vel`：己方卫星速度
- `target_pixel`：目标在图像中的像素坐标 [x, y]
- `angle_deg`：目标相对方向角（度）
- `background_stars`：背景星列表，每项含：
    - `id`：星表唯一id
    - `pixel_x`/`pixel_y`：像素坐标
- `target_orbit_pos`：目标星轨道三维坐标
- `target_orbit_vel`：目标星轨道速度

## asteroid_<t>_result.json 字段说明
- `timestamp`：帧号/时间戳
- `detect_pixel`：算法检测到的像素坐标 [x, y]
- `confidence`：置信度

## 其它说明
- 所有图像分辨率、视场角等参数见主配置文件 `sim_config.toml`。
- 星表唯一id来源于Gaia等星表的`source_id`或`id`字段。
- 可用`build_dataset_index.py`自动生成/更新索引。

