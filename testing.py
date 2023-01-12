import torch
from torch import nn as nn


class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding='same', dilation=1, groups=1, bias=True, scale=10000):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.scale = scale

    def forward(self, x):
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = 0.0
        summed = torch.sum(self.weight.data, dim=(2,3), keepdim=True)/self.scale
        self.weight.data = self.weight.data/summed
        self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = -self.scale

        # # self.weight.data[:, :, self.kernel_size[0]//2, self.kernel_size[1]//2] = -100-torch.sum(self.weight.data, dim=(2,3), keepdim=True)

        # print(self.weight.data.shape)
        # print(self.weight.data.sum())
        # print(self.weight.data)
        return super(CustomConv2d, self).forward(x)




def train(model, data, y):
    opt = torch.optim.Adam(params=model.parameters(), lr=2e-2)
    loss = nn.MSELoss()
    for i in range(10):
        out = model(data)
        error = loss(out.squeeze(), y)
        print(error.item())
        opt.zero_grad()
        error.backward()
        opt.step()







def main():
    x = torch.ones(size=(10, 3, 3, 3))
    x[1, 0, :, :] = 2
    x[1,0,2,2] = 2
    y = torch.zeros(size=(10,))
    model = CustomConv2d(in_channels=3, out_channels=3, stride=1, kernel_size=3, padding=0, scale=100, bias=False)
    out = model(x)
    train(model, x, y)

    # print(x)
    # x[:, :, 3//2, 3//2] = 0
    # summed = torch.sum(x, dim=(2, 3), keepdim=True)/1
    # print(x)
    # print(summed)
    # x = x/summed
    # print(x)
    # x[:, :, 3//2, 3//2] = -100
    # print(x)
    # print(torch.sum(x, dim=(2,3)))

    
    


if __name__ == '__main__':
    main()