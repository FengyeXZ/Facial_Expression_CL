import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datasets.ravdess import MyRAVDESS
from utils.buffer import Buffer
from PIL import Image
from backbone.EfficientNet import mammoth_efficientnet
from utils.center_loss import CenterLoss
import random

# 该区域定义model 后续建立model文件调用
model = mammoth_efficientnet(6, model_name="efficientnet-b0", pretrained=True)
# model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# model = models.resnet50(pretrained=True)

# num_classes = 6  # Number of classes in the data(emotion)
# model.classifier = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(model.classifier[1].in_features, num_classes)
# )
# model.fc = nn.Linear(model.fc.in_features, num_classes)

data_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.6740, 0.4463, 0.3831], [0.2017, 0.1714, 0.1607])
])

train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop(112, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.6740, 0.4463, 0.3831], [0.2017, 0.1714, 0.1607])
])

actor = '04'

# train_data = ImageFolder(root='data/RAVDESS/train/' + actor, transform=train_transforms)
# val_data = ImageFolder(root='data/RAVDESS/val/' + actor, transform=data_transforms)
# test_data = ImageFolder(root='data/RAVDESS/test/' + actor, transform=data_transforms)
train_data = MyRAVDESS(domain_id=2, data_type='train')
val_data = MyRAVDESS(domain_id=2, data_type='val')
test_data = MyRAVDESS(domain_id=2, data_type='test')

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# train_data2 = ImageFolder(root='data/RAVDESS/train/02', transform=train_transforms)
# val_data2 = ImageFolder(root='data/RAVDESS/val/02', transform=data_transforms)
# test_data2 = ImageFolder(root='data/RAVDESS/test/02', transform=data_transforms)
train_data2 = MyRAVDESS(domain_id=4, data_type='train')
val_data2 = MyRAVDESS(domain_id=4, data_type='val')
test_data2 = MyRAVDESS(domain_id=4, data_type='test')

train_loader2 = DataLoader(train_data2, batch_size=64, shuffle=True)
val_loader2 = DataLoader(val_data2, batch_size=64, shuffle=False)
test_loader2 = DataLoader(test_data2, batch_size=64, shuffle=False)

# train_data3 = ImageFolder(root='data/RAVDESS/train/03', transform=train_transforms)
# val_data3 = ImageFolder(root='data/RAVDESS/val/03', transform=data_transforms)
# test_data3 = ImageFolder(root='data/RAVDESS/test/03', transform=data_transforms)
train_data3 = MyRAVDESS(domain_id=3, data_type='train')
val_data3 = MyRAVDESS(domain_id=3, data_type='val')
test_data3 = MyRAVDESS(domain_id=3, data_type='test')

train_loader3 = DataLoader(train_data3, batch_size=64, shuffle=True)
val_loader3 = DataLoader(val_data3, batch_size=64, shuffle=False)
test_loader3 = DataLoader(test_data3, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
model.to(device)

criterion = nn.CrossEntropyLoss()
center_loss = CenterLoss(num_classes=6, feat_dim=1280, use_gpu=True)
# params = list(model.parameters()) + list(center_loss.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)

# Define a simple buffer
buffer_size = 64
buffer = Buffer(buffer_size, device)


def update_buffer(train_set):
    model.eval()  # 设置模型为评估模式以避免影响批正则化等
    criterion.reduction = 'none'  # 计算每个样本的损失，而不是平均损失

    with torch.no_grad():  # 在更新缓冲区时不计算梯度
        for inputs, labels, original in train_set:
            inputs, labels, original = inputs.to(device), labels.to(device), original.to(device)
            outputs = model(inputs)
            losses = criterion(outputs, labels)  # 计算每个样本的损失

            # 找到损失最大的样本
            _, indices_of_max_losses = torch.topk(losses, k=10, largest=True, sorted=False)

            # 选出这些样本和标签
            max_loss_inputs = original[indices_of_max_losses]
            max_loss_labels = labels[indices_of_max_losses]

            # 将这些样本添加到缓冲区中
            buffer.add_data(max_loss_inputs, max_loss_labels)

    model.train()  # 将模型设置回训练模式
    criterion.reduction = 'mean'  # 恢复默认的损失计算方式


def acc_eval(test_set):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, original in test_set:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def model_train(train_set, val_set, epochs=10):
    num_epochs = epochs  # 可以根据需要调整epoch数量

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        for inputs, labels in train_set:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs, features = model(inputs, returnt='all')
            print(features.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            # for param in criterion.parameters():
            #     # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            #     param.grad.data *= (0.5 / 0.01)
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_set)}")

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        print(f"Accuracy on validation set: {acc_eval(val_set)}%")


prob = 1


def train_with_replay(train_set, val_set, buffer, epochs=30):
    global prob
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels, original in train_set:
            inputs, labels = inputs.to(device), labels.to(device)
            mix_inputs = inputs
            mix_labels = labels
            # 从缓冲区中抽取数据进行重播
            if not buffer.is_empty():
                replay_inputs, replay_labels = buffer.get_data(size=32)
                mix_inputs = torch.cat((inputs, replay_inputs), 0)
                mix_labels = torch.cat((labels, replay_labels), 0)

            optimizer.zero_grad()
            optimizer_centloss.zero_grad()

            # 处理当前任务的数据
            outputs, features = model(mix_inputs, returnt='all')
            # print(features.shape)
            loss = criterion(outputs, mix_labels)
            # optimizer.zero_grad()
            # optimizer_centloss.zero_grad()

            # # 从缓冲区中抽取数据进行重播
            # if not buffer.is_empty():
            #     replay_inputs, replay_labels = buffer.get_data(size=16)
            #     replay_inputs, replay_labels = replay_inputs.to(device), replay_labels.to(device)
            #     replay_outputs, replay_features = model(replay_inputs, returnt='all')
            #     replay_loss = criterion(replay_outputs, replay_labels) * 0.1
            #     loss += replay_loss  # 结合当前数据和重播数据的损失

            loss.backward()
            # for param in center_loss.parameters():
            #     param.grad.data *= (1. / 0.1)
            optimizer.step()
            # optimizer_centloss.step()

            running_loss += loss.item()

            chance = random.random()
            if chance < prob:
                buffer.add_data(original, labels)  # 将当前数据添加到缓冲区
            prob *= 0.99

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_set)}")

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        print(f"Accuracy on validation set: {acc_eval(val_set)}%")


acc1 = acc_eval(test_loader)
acc2 = acc_eval(test_loader2)
acc3 = acc_eval(test_loader3)
print(f"Accuracy on test set1(before training): {acc1}%")
print(f"Accuracy on test set2(before training): {acc2}%")
print(f"Accuracy on test set3(before training): {acc3}%")

# model_train(train_loader, val_loader)
train_with_replay(train_loader, val_loader, buffer)

acc1new = acc_eval(test_loader)
acc2new = acc_eval(test_loader2)
acc3new = acc_eval(test_loader3)
def1 = acc1new - acc1
def2 = acc2new - acc2
def3 = acc3new - acc3
print(f"Accuracy on test set1: {acc1new}%", f"Change: {def1}%")
print(f"Accuracy on test set2: {acc2new}%", f"Change: {def2}%")
print(f"Accuracy on test set3: {acc3new}%", f"Change: {def3}%")

# model_train(train_loader2, val_loader2)
train_with_replay(train_loader2, val_loader2, buffer)

acc1 = acc_eval(test_loader)
acc2 = acc_eval(test_loader2)
acc3 = acc_eval(test_loader3)
def1 = acc1 - acc1new
def2 = acc2 - acc2new
def3 = acc3 - acc3new
print(f"Accuracy on test set1: {acc1}%", f"Change: {def1}%")
print(f"Accuracy on test set2: {acc2}%", f"Change: {def2}%")
print(f"Accuracy on test set3: {acc3}%", f"Change: {def3}%")

# model_train(train_loader3, val_loader3)
train_with_replay(train_loader3, val_loader3, buffer)

acc1new = acc_eval(test_loader)
acc2new = acc_eval(test_loader2)
acc3new = acc_eval(test_loader3)
def1 = acc1new - acc1
def2 = acc2new - acc2
def3 = acc3new - acc3
print(f"Accuracy on test set1: {acc1new}%", f"Change: {def1}%")
print(f"Accuracy on test set2: {acc2new}%", f"Change: {def2}%")
print(f"Accuracy on test set3: {acc3new}%", f"Change: {def3}%")
