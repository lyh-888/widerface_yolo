import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# 根目录
root = './data/widerface/'


def xywh2xxyy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return (x1, x2, y1, y2)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def wider2face(phase='val', ignore_small=0):
    data = {}
    current_valid = False
    current_width = 0
    current_height = 0
    current_path = ""

    label_path = os.path.join(root, phase, 'label.txt')
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if '.jpg' in line:
                img_rel_path = line.replace('#', '').strip()
                path = os.path.join(root, 'WIDER_val', 'images', img_rel_path)

                img = cv2.imread(path)
                if img is None:
                    print(f"警告：无法读取图片（路径错误）：{path}")
                    current_valid = False
                    continue

                current_height, current_width, _ = img.shape
                current_path = path
                data[current_path] = []
                current_valid = True
            else:
                if not current_valid or not line:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                try:
                    box = np.array(parts[0:4], dtype=np.float32)
                except ValueError:
                    continue

                if box[2] < ignore_small or box[3] < ignore_small:
                    continue

                box = convert((current_width, current_height), xywh2xxyy(box))
                label = '0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'.format(
                    round(box[0], 4), round(box[1], 4),
                    round(box[2], 4), round(box[3], 4)
                )
                data[current_path].append(label)
    return data


if __name__ == '__main__':
    datas = wider2face('val')

    output_img_dir = os.path.join(root, 'val', 'images')
    output_label_dir = os.path.join(root, 'val', 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for img_path in datas:
        # 处理图片路径和复制
        img_rel_path = os.path.relpath(img_path, os.path.join(root, 'WIDER_val', 'images'))
        dest_img_path = os.path.join(output_img_dir, img_rel_path)
        os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
        shutil.copyfile(img_path, dest_img_path)

        # 处理标注文件路径（关键修复：创建标注文件所在的子目录）
        img_name = os.path.splitext(img_rel_path)[0]
        dest_label_path = os.path.join(output_label_dir, f"{img_name}.txt")
        # 新增：创建标注文件的父目录
        os.makedirs(os.path.dirname(dest_label_path), exist_ok=True)

        # 写入标注文件
        with open(dest_label_path, 'w') as f:
            for label in datas[img_path]:
                f.write(f"{label}\n")

    print(f"转换完成！共处理 {len(datas)} 张图片")
    print(f"图片保存至：{output_img_dir}")
    print(f"标注文件保存至：{output_label_dir}")
