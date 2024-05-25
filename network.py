import torch
from torch import nn

# NOTE: This will be the network architecture.


class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        # relu activation function and max pooling
        relu = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # define the network architecture
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), padding="same"),
            relu,
            pool,
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding="same"),
            relu,
            pool,
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding="same"),
            relu,
            pool,
            nn.Conv2d(64, 128, kernel_size=(5, 5), padding="same"),
            relu,
            pool,
            nn.Conv2d(128, 256, kernel_size=(5, 5), padding="same"),
            relu,
            pool,
            nn.Conv2d(256, 512, kernel_size=(5, 5), padding="same"),
            relu,
            pool,
            nn.Flatten(),
            nn.Linear(8192, nClasses),
        )

    def forward(self, x):
        return self.net(x)
