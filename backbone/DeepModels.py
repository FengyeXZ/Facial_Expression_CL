# Based on https://github.com/rahullabs/FIXR_Public.git
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
from backbone import MammothBackbone
import timm
from torchvision.models.efficientnet import EfficientNet_B0_Weights
import torchvision.models as models


def mammoth_efficientnet(nclasses: int, model_name: str, pretrained=True, train_bbone=False):
    """
    Instantiates a EfficientNet network.
    :param train_bbone:
    :param pretrained:
    :param model_name:
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: EfficientNet network
    """
    # print(model_name)
    efp = timm.create_model('tf_efficientnet_b2_ns', pretrained=True)
    # print(efp)
    print(efp.classifier.in_features)
    # print(efp) to check last later for in_features
    # let's update the pretarined model:
    if not train_bbone:
        for param in efp.parameters():
            param.requires_grad = False
    # print(efp)
    # for tf_efficientnet_b0_ns
    # efp.classifier = nn.Sequential(
    #     nn.Linear(in_features=1280, out_features=512), 
    #     nn.Dropout(p=0.5),
    #     nn.Linear(in_features=512, out_features=nclasses), 
    # )
    # for tf_efficientnet_b2_ns
    efp.classifier = nn.Sequential(
        # nn.Linear(in_features=efp.classifier[1].in_features, out_features=512),
        nn.Linear(in_features=1408, out_features=512),
        nn.Dropout(p=0.5),
        # nn.Linear(in_features=1408, out_features=1024),
        # nn.Dropout(p=0.5),
        # nn.Linear(in_features=1024, out_features=512),
        # nn.Dropout(p=0.5),
        nn.Linear(in_features=512, out_features=nclasses),
    )
    # print(efp)
    param_list = list(efp.parameters())
    # for idx in range(len(param_list)):
    #         if param_list[idx].requires_grad:
    #             print(param_list[idx])
    # for param in efp.parameters():
    #     print(param.grad)
    return efp


def efficientnet_b0(nclasses: int, pretrained=True):
    """
    Instantiates a EfficientNet network.
    :param nclasses: number of output classes
    :return: EfficientNet network
    """
    efp = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    num_classes = nclasses  # Number of classes in the data(emotion)
    print(efp.classifier[1].in_features)
    efp.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(efp.classifier[1].in_features, num_classes)
    )
    print(efp.classifier[1].in_features)

    return efp


def resnet_50(nclasses: int):
    res50 = models.resnet50(pretrained=True)
    num_classes = nclasses
    res50.fc = torch.nn.Linear(res50.fc.in_features, 6)
    return res50
