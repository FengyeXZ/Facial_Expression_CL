from torchvision.datasets import ImageFolder
from datasets.ravdess import MyRAVDESS
from torchvision import transforms

# This code if for the RAVDESS data
my_ravdess = MyRAVDESS(domain_id=1, data_type='test')
print(my_ravdess.class_to_idx())
print(my_ravdess.__len__())
idx = 200
print(my_ravdess.dataset[idx][0], '|', my_ravdess.dataset[idx][1])
print(my_ravdess.__getitem__(idx)[1])

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.6740, 0.4463, 0.3831], [0.2017, 0.1714, 0.1607])
])
test_img = ImageFolder(root='data/RAVDESS/test/01', transform=transform)
print(test_img[idx][0] == my_ravdess.__getitem__(idx)[0])
print(my_ravdess[idx][0] == my_ravdess.__getitem__(idx)[0])

print(test_img.__getitem__(idx)[0] == my_ravdess[idx][0])
