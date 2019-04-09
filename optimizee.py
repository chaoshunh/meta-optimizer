import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    ''' A multi-layer perceptron.'''
    def __init__(self, num_layers=5, channels=32):
        super(MLP, self).__init__()
                         
        self.fc0 = nn.Linear(28*28, channels) # mnist image size:28*28
        self.layers = self._make_layers(num_layers, channels)
        self.output = nn.Linear(channels, 10) # mnist label types: 10

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.layers(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)

    def _make_layers(self, num_layers, channels):
        layers = []
        for l in range(num_layers):
            layers.append(nn.Linear(channels, channels))

        return nn.Sequential(*layers)
                        

class SimpleCNN(nn.Module):
    ''' A simple convolution neural network.'''
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # conv1: 1 input image channel, 6 output channels,
        #        5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2  = nn.Conv2d(8, 16, 5, padding=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*7*7, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
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
    mlp = MLP(num_layers=2, channels=32)
    print(len(list(mlp.parameters())))
    print(len(list(mlp.children())))

    scnn = SimpleCNN()
    x = torch.randn(1,1,28,28)
    print(scnn)
    print(scnn(x))

        

    


    
