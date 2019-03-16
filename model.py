import torch
import torch.nn as nn
import torch.nn.functional as F



class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

if __name__ == "__main__":
    fcnn = FullyConnectedNN()
    print(fcnn)

        

    


    
