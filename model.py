import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
from torchvision import models
import os
import cv2
import numpy as np


class ModelBase(nn.Module):
    def __init__(self, name, created_time):
        super(ModelBase, self).__init__()
        self.name = name
        self.created_time = created_time

    def copy_params(self, state_dict):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

    def boost_params(self, scale=1.0):
        if scale == 1.0:
            return self.state_dict()
        for (name, param) in self.state_dict().items():
            self.state_dict()[name].copy_((scale * param).clone())
        return self.state_dict()

    # self - x
    def sub_params(self, x):
        own_state = self.state_dict()
        for (name, param) in x.items():
            if name in own_state:
                own_state[name].copy_(own_state[name] - param)

    # self + x
    def add_params(self, x):
        a = self.state_dict()
        for (name, param) in x.items():
            if name in a:
                a[name].copy_(a[name] + param)



constlayer = dict(outch=1, ks=5, scale=10000)

class ConstConv(ModelBase):
    """
    doc
    """
    def __init__(self, lcnf: dict=constlayer, name='constlayer', created_time=None, num_cls=10):
        super().__init__(name=name, created_time=created_time)
        self.lcnf = lcnf
        self.num_cls = num_cls
        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[lcnf['outch'], 1, lcnf['ks'], lcnf['ks']]), requires_grad=True)

        resnet_weight = models.ResNet50_Weights.DEFAULT
        self.base_model = models.resnet50(weights=resnet_weight)
        self.base_model.fc = nn.Linear(in_features=2048, out_features=num_cls)

    def normalize(self):
        cntrpxl = int(self.lcnf['ks']/2)
        centeral_pixels = (self.const_weight[:, 0, cntrpxl, cntrpxl])
        for i in range(self.lcnf['outch']):
            sumed = (self.const_weight.data[i].sum() - centeral_pixels[i])/self.lcnf['scale']
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, cntrpxl, cntrpxl] = -self.lcnf['scale']

    def forward(self, x):
        self.normalize()
        noise = F.conv2d(x[:, 0:1, :, :], self.const_weight, padding='same')
        xn = torch.cat((noise, x[:, 1:3, :, :]), dim=1)
        x = self.base_model(xn)
        
        return x , noise



def main():
    x = torch.randn(size=(3, 3, 224, 224))

    model = ConstConv(num_cls=6)
    yhat, noise = model(x)
    print(yhat)
    print(noise.shape)
    noise = noise.detach().numpy()
    

    # for noisech in noise:
    #     cv2.imshow('noise', noisech[0])
    #     cv2.waitKey(0)

    # for noisech in noisecoord:
    #     noisech = np.reshape(noisech, newshape=(224, 224, 3))
    #     cv2.imshow('noisecoord', noisech)
    #     cv2.waitKey(0)


if __name__ == '__main__':
    main()