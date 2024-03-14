# This code if for the MEAD data
import os
from typing import Tuple
from PIL import Image
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets.transforms.denormalization import DeNormalize
from torchvision.datasets import ImageFolder
from datasets.utils.continual_dataset import ContinualDataset
from backbone.EfficientNet import mammoth_efficientnet
from backbone.DeepModels import efficientnet_b0, resnet_50
from utils.center_loss import CenterLoss


def store_mead_dataset(domain_id, transform, setting):
    train_dataset = MyMEAD(domain_id, verbose=True)
    test_dataset = MyMEAD(domain_id=domain_id, data_type='val', verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, drop_last=True)

    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    return train_loader, test_loader


def store_test_mead_dataset(domain_id, setting):
    test_dataset = MyMEAD(domain_id=domain_id, data_type='val', verbose=True)

    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False)

    setting.test_loader = test_loader
    return test_loader


class MyMEAD(Dataset):
    def __init__(self, domain_id: int, root='data/MEAD', data_type='train', img_size=112, transform=None,
                 target_transform=None, verbose=False) -> None:
        self.data_path = os.path.join(root, str(data_type), str(domain_id).zfill(2))
        self.img_size = img_size
        self.data_type = data_type
        self.transform = transform

        self.__process__()

    def __getitem__(self, idx: int) -> Tuple[type(Image), int, type(Image)]:
        mean, std = (0.6740, 0.4463, 0.3831), (0.2017, 0.1714, 0.1607)
        img_size = self.img_size
        resize_img = 128

        if self.data_type == 'train':
            transform = transforms.Compose([
                transforms.Resize((resize_img, resize_img)),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        not_aug_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        img, target = self.dataset[idx][0], self.dataset[idx][1]
        original_img = img.copy()

        img = transform(img)
        not_aug_img = not_aug_transform(original_img)

        return img, target, not_aug_img

    def __len__(self):
        return len(self.dataset)

    def __process__(self):
        if 224 >= self.img_size >= 32:
            self.dataset = self.generate_tensor(img_size=self.img_size)
        else:
            print("{} dataset does not matches with dataset requirement")

    def generate_tensor(self, img_size=244, normalize=False):
        return ImageFolder(self.data_path)

    def class_to_idx(self):
        return self.dataset.class_to_idx


class MEAD(ContinualDataset):
    NAME = 'ravdess'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 6
    N_TASKS = 24  # Number of Domains
    MEAN, STD = (0.6740, 0.4463, 0.3831), (0.2017, 0.1714, 0.1607)
    IMG_SIZE = 112
    IMG_RESIZE = 128
    TRANSFORM = transforms.Compose([
        transforms.Resize(IMG_RESIZE),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def get_nonperm_data_loaders(self):
        transform = transforms.Compose((transforms.ToTensor(),))
        train, test = store_mead_dataset(1, transform, self)
        return train, test

    def data_loader_with_did(self, did):
        domain_id = self.get_did(domain_id=did)
        transform = transforms.Compose((transforms.ToTensor(),))
        train, test = store_mead_dataset(domain_id, transform, self)
        return train, test

    def test_data_loader_with_did(self, did):
        domain_id = self.get_did(domain_id=did)
        transform = transforms.Compose((transforms.ToTensor(),))
        test = store_test_mead_dataset(domain_id, self)
        return test

    def get_did(self, domain_id):
        DOMAIN_ID = domain_id
        return DOMAIN_ID

    @staticmethod
    def get_backbone(args=None):
        return mammoth_efficientnet(MEAD.N_CLASSES_PER_TASK, model_name="efficientnet-b0", pretrained=True)
        # return efficientnet_b0(RAVDESS.N_CLASSES_PER_TASK)
        # return resnet_50(RAVDESS.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            MEAD.TRANSFORM
        ])

    @staticmethod
    def get_loss():
        return nn.CrossEntropyLoss()

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(MEAD.MEAN, MEAD.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(MEAD.MEAN, MEAD.STD)


def main():
    r = MEAD(verbose=True)


if __name__ == '__main__':
    main()
