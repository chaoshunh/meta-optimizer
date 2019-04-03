import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    ''' A multi-layer perceptron.'''
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 32) # mnist image size:28*28
        self.fc2 = nn.Linear(32, 10) # mnist label types: 10

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Simple_CNN(nn.Module):
    ''' A simple convolution neural network.'''
    def __init__(self):
        super(Simple_CNN, self).__init__()
        # conv1: 1 input image channel, 6 output channels,
        #        5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*14*14, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DenseNet(nn.Module):
    pass

if __name__ == "__main__":
    mlp = MLP()
    scnn = Simple_CNN()
    x = torch.randn(1, 1, 28, 28)
    print(mlp(x))
    print(scnn(x))

        

    


    
