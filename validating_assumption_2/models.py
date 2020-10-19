import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nc):
        super(Net, self).__init__()

        if nc == 1:
            d = 4 * 4 * 50
        elif nc == 3:
            d = 5 * 5 * 50

        self.d = d

        self.conv1 = nn.Conv2d(nc, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(d, 300)
        self.fc2 = nn.Linear(300, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.d)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, nc, h, width):
        super(DenseNet, self).__init__()
        self.nc = nc
        self.h = h
        self.fc1 = nn.Linear(nc * h * h, width)
        self.fc2 = nn.Linear(width, 10)
        with torch.no_grad():
            self.fc1.weight.div_(torch.norm(self.fc1.weight, dim=1, keepdim=True))
            self.fc2.weight.div_(torch.norm(self.fc2.weight, dim=1, keepdim=True))

    def forward(self, x):

        x = x.view(-1, self.h**2 * self.nc)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


