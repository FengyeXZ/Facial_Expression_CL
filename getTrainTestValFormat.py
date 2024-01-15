import os
import shutil
import random
from tqdm import tqdm


def split_and_rename_data(source_folder, train_folder, test_folder, val_folder, train_size=0.7, val_size=0.15):
    # 创建训练、测试和验证目录
    for folder in [train_folder, test_folder, val_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 遍历每个类别的情绪目录
    for domain in tqdm(os.listdir(source_folder)):
        domain_folder = os.path.join(source_folder, domain)
        for emotion in os.listdir(domain_folder):
            emotion_folder = os.path.join(domain_folder, emotion)

            # 只处理文件夹
            if os.path.isdir(emotion_folder):
                images = os.listdir(emotion_folder)
                random.shuffle(images)  # 打乱图像顺序

                # 计算训练、测试和验证集的分割点
                train_end = int(len(images) * train_size)
                val_end = train_end + int(len(images) * val_size)

                # 复制和重命名图像到对应的训练、测试和验证目录
                for i, img in enumerate(images):
                    src_path = os.path.join(emotion_folder, img)

                    if i < train_end:
                        dest_folder = os.path.join(train_folder, domain, emotion)
                        img_index = i + 1
                    elif i < val_end:
                        dest_folder = os.path.join(val_folder, domain, emotion)
                        img_index = i - train_end + 1
                    else:
                        dest_folder = os.path.join(test_folder, domain, emotion)
                        img_index = i - val_end + 1

                    # 确保目标目录存在
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)

                    dest_path = os.path.join(dest_folder, f'{img_index}.png')
                    shutil.copy(src_path, dest_path)


# 主要配置
source_folder = 'expData-processed'  # 原始数据集目录
train_folder = 'dataset/train'  # 训练集目录
test_folder = 'dataset/test'  # 测试集目录
val_folder = 'dataset/val'  # 验证集目录

# 执行数据集分割和重命名
split_and_rename_data(source_folder, train_folder, test_folder, val_folder)
