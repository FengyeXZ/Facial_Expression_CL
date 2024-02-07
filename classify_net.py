import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_B0_Weights, EfficientNet_B2_Weights, EfficientNet_B3_Weights, \
    EfficientNet_B5_Weights, EfficientNet_B6_Weights
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


# 该区域定义model 后续建立model文件调用
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

num_classes = 6  # Number of classes in the data(emotion)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

data_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

actor = '01'

train_data = ImageFolder(root='data/train/' + actor, transform=data_transforms)
val_data = ImageFolder(root='data/val/' + actor, transform=data_transforms)
test_data = ImageFolder(root='data/test/' + actor, transform=data_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

train_data2 = ImageFolder(root='data/train/02', transform=data_transforms)
val_data2 = ImageFolder(root='data/val/02', transform=data_transforms)
test_data2 = ImageFolder(root='data/test/02', transform=data_transforms)

train_loader2 = DataLoader(train_data2, batch_size=32, shuffle=True)
val_loader2 = DataLoader(val_data2, batch_size=32, shuffle=False)
test_loader2 = DataLoader(test_data2, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def acc_eval(test_set):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_set:
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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_set)}")

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        print(f"Accuracy on validation set: {acc_eval(val_set)}%")


print(f"Accuracy on test set1(before training): {acc_eval(test_loader)}%")
print(f"Accuracy on test set2(before training): {acc_eval(test_loader2)}%")

model_train(train_loader, val_loader)

print(f"Accuracy on test set1: {acc_eval(test_loader)}%")
print(f"Accuracy on test set2: {acc_eval(test_loader2)}%")

model_train(train_loader2, val_loader2)

print(f"Accuracy on test set1: {acc_eval(test_loader)}%")
print(f"Accuracy on test set2: {acc_eval(test_loader2)}%")
