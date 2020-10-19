import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nc):
        super(Net, self).__init__()

        if nc == 1: d = 40
        elif nc == 3: d = 160

        self.d = d

        self.conv1 = nn.Conv2d(nc, 20, 10, 2)
        self.conv2 = nn.Conv2d(20, 40, 10, 2)
        self.fc1 = nn.Linear(d, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x


class ShallowNet(nn.Module):
    def __init__(self, nc, h):
        super(ShallowNet, self).__init__()
        self.nc = nc
        self.h = h
        self.fc1 = nn.Linear(nc * h * h, 100)
        self.fc2 = nn.Linear(100, 10)
        with torch.no_grad():
            self.fc1.weight.div_(torch.norm(self.fc1.weight, dim=1, keepdim=True))
            self.fc2.weight.div_(torch.norm(self.fc2.weight, dim=1, keepdim=True))

    def forward(self, x):

        x = x.view(-1, self.h**2 * self.nc)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


