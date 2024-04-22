import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import DBSCAN
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的模型
model = models.resnet50(pretrained=True)
model.eval()  # 设置为评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0)  # 增加批次维度
    return img_tensor


def extract_features(directory):
    features = []
    labels = []

    for domain_folder in sorted(os.listdir(directory)):
        domain_path = os.path.join(directory, domain_folder)
        for emotion_folder in sorted(os.listdir(domain_path)):
            emotion_path = os.path.join(domain_path, emotion_folder)
            for image_name in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, image_name)
                image_tensor = load_and_preprocess_image(image_path)

                # 提取特征
                with torch.no_grad():
                    feature = model(image_tensor).squeeze(0).cpu().numpy()  # 移动到CPU并转为NumPy数组
                features.append(feature)
                labels.append(emotion_folder)  # 使用文件夹名称作为标签

    return features, labels


def apply_clustering(features):
    clustering = DBSCAN(eps=3, min_samples=2).fit(features)
    return clustering.labels_
# 可以添加代码来分析聚类结果，比如统计每个类别的样本数量，查看噪声点等


def visualize_clusters(features, labels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of feature clusters')
    plt.xlabel('t-SNE axis 1')
    plt.ylabel('t-SNE axis 2')
    plt.show()


def clean_data(features, labels, cluster_labels):
    noise_indices = cluster_labels == -1
    clean_features = [f for i, f in enumerate(features) if not noise_indices[i]]
    clean_labels = [l for i, l in enumerate(labels) if not noise_indices[i]]
    return clean_features, clean_labels


# 假设你已经提取了特征和标签
features, labels = extract_features('dataTmp/data')
features_np = np.array(features)
cluster_labels = apply_clustering(features_np)

# 可视化聚类结果
visualize_clusters(features_np, cluster_labels)

# 清洗数据
clean_features, clean_labels = clean_data(features_np, labels, cluster_labels)

# 输出清洗后数据的信息
print(f"Original data count: {len(features_np)}")
print(f"Clean data count: {len(clean_features)}")


