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
            # 识别图片路径行（包含.jpg）
            if '.jpg' in line:
                if isFirst is True:
                    isFirst = False
                else:
                    # 保存上一张图片的标注
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                # 处理图片路径
                path = line.lstrip('# ').strip()
                parent_dir = os.path.dirname(txt_path)
                # 拼接正确的图片路径
                path = os.path.normpath(os.path.join(parent_dir, '../WIDER_train/images', path))
                self.imgs_path.append(path)
            else:
                # 处理标注行
                try:
                    # 人脸数量行（纯数字）
                    num_faces = int(line)
                    continue
                except ValueError:
                    # 标注数据行
                    line_split = line.split()
                    # 检查标注行是否有10个元素（WIDER Face格式）
                    if len(line_split) != 10:
                        print(f"警告：标注行元素数量错误（需10个，实际{len(line_split)}个），已跳过")
                        continue
                    label = [float(x) for x in line_split]
                    labels.append(label)

        # 添加最后一张图片的标注
        if len(labels) > 0:
            self.words.append(labels.copy())

        # 验证图片和标注数量是否匹配
        assert len(self.imgs_path) == len(self.words), \
            f"图片数量（{len(self.imgs_path)}）与标注数量（{len(self.words)}）不匹配"

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 5))  # 仅存储边界框信息和类别
        if len(labels) == 0:
            return annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))
            # 只使用前4个边界框参数（x1, y1, w, h）
            x1 = label[0]
            y1 = label[1]
            w = label[2]
            h = label[3]

            # 计算YOLO格式的中心点和宽高（归一化）
            annotation[0, 0] = (x1 + w / 2) / width  # 中心点x坐标
            annotation[0, 1] = (y1 + h / 2) / height  # 中心点y坐标
            annotation[0, 2] = w / width  # 宽度
            annotation[0, 3] = h / height  # 高度
            annotation[0, 4] = 0  # 类别：人脸

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
        print('示例: python data/train2yolo.py ./data/widerface/train ./data/widerface/train/output')
        exit(1)

    original_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else './data/widerface/train/output'

    # 创建输出目录
    os.makedirs(save_path, exist_ok=True)

    # 检查标注文件是否存在
    label_path = os.path.join(original_path, 'label.txt')
    if not os.path.isfile(label_path):
        print(f'错误：找不到标注文件 {label_path}')
        exit(1)

    # 加载数据集
    dataset = WiderFaceDetection(label_path)
    print(f"发现 {len(dataset)} 张图片和对应的标注")

    # 转换并保存为YOLO格式
    for i in range(len(dataset)):
        img_path = dataset.imgs_path[i]
        print(f"处理第{i + 1}/{len(dataset)}张: {img_path}")

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}，已跳过")
            continue

        # 获取标注
        if i >= len(dataset.words):
            print(f"警告：图片 {img_path} 无对应标注，已跳过")
            continue
        labels = dataset.words[i]

        # 保存图片
        img_name = os.path.basename(img_path)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, img)

        # 保存标注文件（YOLO格式）
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        save_txt_path = os.path.join(save_path, txt_name)

        with open(save_txt_path, 'w') as f:
            height, width, _ = img.shape
            for label in labels:
                # 只使用前4个边界框参数
                x1, y1, w, h = label[0], label[1], label[2], label[3]
                # 计算YOLO格式的中心点和宽高（归一化）
                cx = (x1 + w / 2) / width
                cy = (y1 + h / 2) / height
                bw = w / width
                bh = h / height
                # 写入文件（类别为0，坐标保留6位小数）
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("转换完成！结果保存在:", save_path)