import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# 假设您已经定义了 MyRAVDESS 类来加载数据
class RAVDESSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): RAVDESS 数据集目录，包含 train/val/test 子目录。
            transform (callable, optional): 应用于图像的可选转换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # 存储图像路径和标签
        self._load_dataset()

    def _load_dataset(self):
        emotion_to_label = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5
        }
        # 假设标签是基于目录结构：root/train/01/.../1img.png
        for split in ['train', 'val']:
            split_dir = os.path.join(self.root_dir, split)
            for domain in os.listdir(split_dir):
                domain_dir = os.path.join(split_dir, domain)
                for emotion in os.listdir(domain_dir):
                    emotion_dir = os.path.join(domain_dir, emotion)
                    for img_name in os.listdir(emotion_dir):
                        img_path = os.path.join(emotion_dir, img_name)
                        label = emotion_to_label[emotion]  # 使用情感名称获取标签
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')  # 确保为三通道

        if self.transform:
            image = self.transform(image)

        return image, label


def calculate_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def main():
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])

    ravdess_dataset = RAVDESSDataset(root_dir='data/MEAD', transform=transform)

    loader = DataLoader(ravdess_dataset, batch_size=64, shuffle=False, num_workers=16)

    mean, std = calculate_mean_std(loader)
    print(f'Mean: {mean}')
    print(f'Std: {std}')


if __name__ == "__main__":
    main()
