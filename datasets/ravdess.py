# This code if for the RAVDESS data
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class MyRAVDESS(Dataset):
    def __init__(self, domain_id, root='data/ravdess', data_type='train', img_size=112, transform=None,
                 target_transform=None, download=False) -> None:
        self.data_path = os.path.join(root, str(data_type), domain_id)
        self.img_size = img_size
        self.data_type = data_type

        self.__process__()

    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        return len(self.dataset)

    def __process__(self):
        if 224 >= self.img_size >= 32:
            self.dataset = self.generate_tensor(img_size=self.img_size)
        else:
            print("{} dataset does not matches with dataset requirement")

    def generate_tensor(self, img_size=244, normalize=False):
        return ImageFolder(self.data_path)
