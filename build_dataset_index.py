# 自动生成数据集索引脚本
# 扫描res目录下所有图片和真值，生成dataset_index.json
import os
import json

def build_index(res_dir='res', out_file='dataset_index.json'):
    entries = []
    for fname in os.listdir(res_dir):
        if fname.endswith('.png'):
            base = fname[:-4]
            img_path = os.path.join(res_dir, fname)
            truth_path = os.path.join(res_dir, base + '_truth.json')
            if os.path.exists(truth_path):
                with open(truth_path, 'r') as f:
                    truth = json.load(f)
                entries.append({
                    'image': img_path,
                    'truth': truth_path,
                    'timestamp': truth.get('timestamp'),
                    'target_pixel': truth.get('target_pixel'),
                    'vel': truth.get('vel'),
                    'angle_deg': truth.get('angle_deg'),
                    'background_stars': truth.get('background_stars')
                })
    # 按时间戳排序
    entries.sort(key=lambda x: x['timestamp'])
    with open(os.path.join(res_dir, out_file), 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"[INFO] 索引已生成: {os.path.join(res_dir, out_file)} 共{len(entries)}条")

if __name__ == '__main__':
    build_index()
