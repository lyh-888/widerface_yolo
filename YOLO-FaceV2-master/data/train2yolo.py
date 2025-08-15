import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip().strip()
            if not line:
                continue
            if '.jpg' in line:
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                # 路径处理：指向实际图片目录
                path = line.lstrip('# ').strip()
                parent_dir = os.path.dirname(txt_path)
                path = os.path.normpath(os.path.join(parent_dir, '../WIDER_train/images', path))
                self.imgs_path.append(path)
            else:
                # 处理标注行（WIDER Face只有10个元素）
                try:
                    num_faces = int(line)
                    continue
                except ValueError:
                    line_split = line.split()
                    # 检查标注行是否有10个元素（WIDER Face标准格式）
                    if len(line_split) != 10:
                        print(f"警告：标注行元素数量错误（需10个，实际{len(line_split)}个），跳过")
                        continue
                    label = [float(x) for x in line_split]
                    labels.append(label)

        # 添加最后一张图片的标注
        if len(labels) > 0:
            self.words.append(labels.copy())

        # 验证数量匹配
        assert len(self.imgs_path) == len(self.words), \
            f"图片数（{len(self.imgs_path)}）与标注数（{len(self.words)}）不匹配"

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        if img is None:
            raise ValueError(f"无法读取图片: {self.imgs_path[index]}")
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 5))  # 只保留边界框+类别
        if len(labels) == 0:
            return annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))
            # 只处理边界框（前4个元素）
            x1 = label[0]
            y1 = label[1]
            w = label[2]
            h = label[3]
            x2 = x1 + w
            y2 = y1 + h

            # 边界框归一化（YOLO格式：cx, cy, w, h）
            annotation[0, 0] = (x1 + x2) / 2 / width  # cx
            annotation[0, 1] = (y1 + y2) / 2 / height  # cy
            annotation[0, 2] = w / width  # 宽
            annotation[0, 3] = h / height  # 高
            annotation[0, 4] = 0  # 类别：face

            annotations = np.append(annotations, annotation, axis=0)

        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('使用方法: python data/train2yolo.py 标注目录 输出目录')
        exit(1)

    original_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else './data/widerface/train'

    # 创建输出目录（分别创建images和labels子目录）
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels'), exist_ok=True)

    # 检查标注文件
    label_path = os.path.join(original_path, 'label.txt')
    if not os.path.isfile(label_path):
        print(f'找不到标注文件: {label_path}')
        exit(1)

    # 加载数据
    dataset = WiderFaceDetection(label_path)
    print(f"总数据量: {len(dataset)} 张图片")

    # 转换并保存
    for i in range(len(dataset)):
        img_path = dataset.imgs_path[i]
        print(f"处理第{i}张: {img_path}")

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的图片: {img_path}")
            continue

        # 获取标注
        if i >= len(dataset.words):
            print(f"跳过无标注的图片: {img_path}")
            continue
        labels = dataset.words[i]

        # 保存图片到images子目录
        img_name = os.path.basename(img_path)
        save_img_path = os.path.join(save_path, 'images', img_name)  # 修改此处路径
        cv2.imwrite(save_img_path, img)

        # 保存标注文件到labels子目录
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        save_txt_path = os.path.join(save_path, 'labels', txt_name)  # 修改此处路径

        with open(save_txt_path, 'w') as f:
            height, width, _ = img.shape
            for label in labels:
                x1, y1, w, h = label[0], label[1], label[2], label[3]
                cx = (x1 + w / 2) / width
                cy = (y1 + h / 2) / height
                bw = w / width
                bh = h / height
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("转换完成！")