import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
from torchvision import models
import os
import cv2
import numpy as np


class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1, bias=False, scale=10000):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.scale = scale

    def forward(self, x):
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0.0
        summed = torch.sum(self.weight.data, dim=(2,3), keepdim=True)/self.scale
        self.weight.data = self.weight.data/summed
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = -self.scale
        return super(CustomConv2d, self).forward(x)


class ConstConv(nn.Module):
    """
    doc
    """
    def __init__(self, num_cls=10):
        super().__init__()
        self.num_cls = num_cls
        self.constconv = CustomConv2d(in_channels=3, out_channels=3, kernel_size=5, padding='same', scale=10000)
        resnet_weight = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=resnet_weight)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=num_cls)


    def forward(self, x):
        xn = self.constconv(x)
        x = self.base_model(xn)
        return x , xn



def main():
    x = torch.randn(size=(1, 3, 224, 224))
    img = cv2.imread('/Users/hamzeasadi/python/resnetsource/data/liebherr/liebherrdataset/liebherrdatasettest/GantryTravel/folder_5_img-0_patch_13.png')
    x = torch.from_numpy(img)
    x = x.permute(2,0,1).type(torch.float32).unsqueeze(dim=0)
    print(x.shape)
    model = ConstConv(num_cls=6)
    yhat, noise = model(x)
    print(yhat)
    print(noise.shape)
    noise = noise.detach().squeeze().numpy()

    for i in range(3):
        cv2.imshow('res', noise[i])
        cv2.waitKey(0)
    



if __name__ == '__main__':
    main()