import torch.nn as nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Dropout

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.C1 = Sequential(Conv2d(in_channels=1, out_channels=32, kernel_size=3), BatchNorm2d(32), ReLU(), MaxPool2d(5, 2))
        self.C2 = Sequential(Conv2d(32, 64, 5), BatchNorm2d(64), ReLU(), MaxPool2d(3, 2))
        self.C3 = Sequential(Conv2d(64, 128, 3), BatchNorm2d(128), ReLU(), MaxPool2d(2, 2))
        self.C4 = Sequential(Conv2d(128, 128, 3), BatchNorm2d(128), ReLU())
        self.C5 = Sequential(Conv2d(128, 32, 1), BatchNorm2d(32), ReLU())
        self.C6 = Sequential(Conv2d(32, 4, 1), BatchNorm2d(4), ReLU())

        # Fully Connected Layers
        self.FC = Sequential(Linear(1600, 256), ReLU(), Dropout(0.5), Linear(256, 64), ReLU(), Linear(64, 16), ReLU(), Linear(16, 3))

    def forward(self, x):
        
        x = self.C6(self.C5(self.C4(self.C3(self.C2(self.C1(x))))))
        # x = self.C1(x)
        # x = self.C2(x)
        # x = self.C3(x)
        # x = self.C4(x)
        # x = self.C5(x)
        # x = self.C6(x)
        B, C, H, W = x.shape
        x = x.view(-1, C * H * W)
        x = self.FC(x)
        return x
