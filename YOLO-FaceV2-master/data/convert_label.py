import os
from tqdm import tqdm


def convert_5to15_column_labels(labels_dir):
    """
    将只有5列的标签文件转换为15列格式
    原始格式：class_id x_center y_center width height
    目标格式：class_id x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
    其中x1-y5均用-1填充（表示无关键点信息）
    """
    # 遍历标签目录下的所有txt文件
    for filename in tqdm(os.listdir(labels_dir), desc="转换标签文件"):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)

            # 读取原始标签内容
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 处理每一行
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 按空格分割列
                parts = line.split()
                if len(parts) != 5:
                    print(f"警告：文件 {filename} 中存在不符合5列格式的行：{line}，已跳过")
                    continue

                # 补充10个-1作为关键点坐标
                # 格式：class_id x_center y_center width height x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
                new_line = parts + ['-1'] * 10
                new_lines.append(' '.join(new_line))

            # 写回处理后的内容
            with open(file_path, 'w') as f:
                f.write('\n'.join(new_lines))

    print("标签转换完成！所有文件已转为15列格式")


if __name__ == "__main__":
    # 请修改为你的标签文件所在目录
    labels_directory = r"D:\学年论文\YOLO-FaceV2-master\YOLO-FaceV2-master\data\widerface\val\labels"

    # 检查目录是否存在
    if not os.path.exists(labels_directory):
        print(f"错误：目录 {labels_directory} 不存在，请检查路径")
    else:
        convert_5to15_column_labels(labels_directory)
        print(f"已处理目录：{labels_directory}")
