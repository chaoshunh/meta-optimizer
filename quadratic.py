import numpy as np
import torch
import torch.nn as nn

class QuadraticLoss:
    def __init__(self):
        super().__init__()
        self.W = torch.randn(10, 10)
        self.y = torch.randn(10)

    def get_loss(self, theta):
        return torch.sum((torch.matmul(self.W, theta) - self.y)**2)

class QuadOptimizee(nn.Module):
    def __init__(self, theta=None):
        super().__init__()
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(10))
        else:
            self.theta = theta

    def forward(self, quadratic):
        return quadratic.get_loss(self.theta)

    def all_named_paramsters(self):
        return [('theta', self.theta)]

def SGD(gradients, lr=0.001):
    return -gradients*lr

def train():
    quadratic = QuadraticLoss()
    optimizee = QuadOptimizee()
    result_params = {}
    for i in range(100):
        loss = optimizee(quadratic)
        loss.backward()

        for name, param in optimizee.all_named_paramsters():
            param_size = int(np.prod(param.size()))
            grads = param.grad.view(param_size, 1)
          #  updates = SGD(grads)
          
            result_params[name] = param+updates.view(*param.size())*1.0
            result_params[name].retain_grad()
   
        optimizee = QuadOptimizee(**result_params)
        print('loss is ', loss.item())
    
if __name__ == '__main__':
    print('training...')
    train()
        
    
