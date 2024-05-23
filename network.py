import torch
from torch import nn

# NOTE: This will be the network architecture.


class Net(nn.Module):
    def __init__(self, nClasses):
        super().__init__()

        # TODO: Implement module constructor.
        # Define network architecture as needed
        # Input imags will be 3 channels 256x256 pixels.
        # Output must be a nClasses Tensor.

        # new_network

        
        # relu activation function and max pooling 
        relu = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        # define the network architecture
        self.net = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(5,5), padding="same"), relu, pool,
        nn.Conv2d(16, 32, kernel_size=(5,5), padding="same"), relu, pool,
        nn.Conv2d(32, 64, kernel_size=(5,5), padding="same"), relu, pool,
        nn.Conv2d(64, 128, kernel_size=(5,5), padding="same"), relu, pool,
        nn.Conv2d(128, 256, kernel_size=(5,5), padding="same"), relu, pool,
        nn.Conv2d(256, 512, kernel_size=(5,5), padding="same"), relu, pool,
        nn.Flatten(),
        nn.Linear(8192, nClasses)
        )

        
 

    def forward(self, x):
        # TODO:
        # Implement forward pass
        # x is a BATCH_SIZEx3x256x256 Tensor
        # return value must be a BATCH_SIZExN_CLASSES Tensor
        return self.net(x)
