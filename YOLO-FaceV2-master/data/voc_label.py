import os
import cv2

# 中间标注目录（convert.py的输出）
anno_dir = 'data/widerface/annotations'
# 原始图片目录
img_dir = 'data/widerface/WIDER_train/images'
# YOLO格式标注输出目录
yolo_anno_dir = 'data/widerface/yolo_annotations'
os.makedirs(yolo_anno_dir, exist_ok=True)

for txt_file in os.listdir(anno_dir):
    if not txt_file.endswith('.txt'):
        continue
    # 对应图片路径（替换标注文件名中的下划线为斜杠）
    img_name = txt_file.replace('_', '/').replace('.txt', '.jpg')
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        continue  # 跳过不存在的图片

    # 读取图片尺寸
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # 转换标注
    with open(os.path.join(anno_dir, txt_file), 'r') as f:
        lines = f.readlines()
    with open(os.path.join(yolo_anno_dir, txt_file), 'w') as f_out:
        for line in lines:
            x1, y1, bw, bh = map(int, line.strip().split())
            # 计算YOLO格式：中心x、中心y、宽、高（归一化）
            cx = (x1 + bw / 2) / w
            cy = (y1 + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h
            f_out.write(f"0 {cx:.6f} {cy:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")  # 0表示人脸类别

print(f"YOLO format annotations saved to {yolo_anno_dir}")