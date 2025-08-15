import os
import shutil

# 原始标注路径（根据实际下载位置修改）
anno_path = 'YOLO-FaceV2-master/data/widerface/wider_face_split/wider_face_train_bbx_gt.txt'
# 输出中间标注目录
out_dir = 'data/widerface/annotations'
os.makedirs(out_dir, exist_ok=True)

with open(anno_path, 'r') as f:
    lines = f.readlines()

idx = 0
while idx < len(lines):
    line = lines[idx].strip()
    if not line:
        idx += 1
        continue
    # 图片路径
    img_path = line
    idx += 1
    # 人脸数量
    num_faces = int(lines[idx].strip())
    idx += 1
    # 保存标注的txt文件
    txt_path = os.path.join(out_dir, img_path.replace('/', '_').replace('.jpg', '.txt'))
    with open(txt_path, 'w') as ft:
        for _ in range(num_faces):
            if idx >= len(lines):
                break
            # 原始标注格式：x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            parts = lines[idx].strip().split()
            x1, y1, w, h = parts[:4]
            # 只保留边界框（YOLO格式需要归一化，后续处理）
            ft.write(f"{x1} {y1} {w} {h}\n")
            idx += 1

print(f"Converted annotations saved to {out_dir}")