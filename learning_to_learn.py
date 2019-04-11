import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm

DIM, batch_size = 10, 128
D_H, num_layers = 20, 2

def detach_var(v):
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var

def training(optimizer, meta_optim, TargetLoss, Optimizee,
             unroll=20, optim_it=100,
              out_mul=1.0, should_train=True):
    '''Calcate the total sum loss of optimizer on training optimizee.'''
    if should_train:
        optimizer.train()
    else:
        optimizer.eval()
        unroll = 1
    
    target = TargetLoss(training=should_train)
    optimizee = Optimizee()
    n_params = 0
    for param in optimizee.parameters():
        n_params += int(np.prod(param.size()))
        
    hidden_states = [Variable(torch.zeros(n_params, optimizer.D_H)) for _ in range(2)]
    cell_states = [Variable(torch.zeros(n_params, optimizer.D_H)) for _ in range(2)]
    all_losses_ever = []
    
    if should_train:
        meta_optim.zero_grad()
    all_losses = None
    
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.item())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [Variable(torch.zeros(n_params, optimizer.D_H)) for _ in range(2)]
        cell_states2 = [Variable(torch.zeros(n_params, optimizer.D_H)) for _ in range(2)]
        for name, param in optimizee.all_named_parameters():
            param_size = int(np.prod(param.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(param.grad.view(param_size, 1))
            updates, new_hidden, new_cell = optimizer(
                gradients,
                [h[offset:offset+param_size] for h in hidden_states],
                [c[offset:offset+param_size] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+param_size] = new_hidden[i]
                cell_states2[i][offset:offset+param_size] = new_cell[i]
            result_params[name] = param + updates.view(*param.size()) * out_mul
            result_params[name].retain_grad()
            
        if iteration % unroll == 0:
            if should_train:
                meta_optim.zero_grad()
                all_losses.backward()
                meta_optim.step()
                
            all_losses = None
                        
            optimizee = Optimizee(**{k: detach_var(v) for k, v in result_params.items()})
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            
        else:
            optimizee = Optimizee(**result_params)
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
            
    return all_losses_ever


def train_optimizer(TargetLoss, Optimizee, preproc=False, n_epochs=2, n_tests=10, lr=0.001):
    ''' Train the optimizer .'''
    optimizer = Optimizer(preproc=preproc)
    meta_optim = optim.Adam(optimizer.parameters(), lr=lr)
    
    best_net = None
    best_loss = 100000000000000000
    
    for epoch in range(n_epochs):
        for _ in range(10): # iterations
            training(optimizer, meta_optim, TargetLoss, Optimizee,
                   should_train=True)
        
        loss = (np.mean([
            np.sum(training(optimizer, meta_optim, TargetLoss, Optimizee,
                            n_epochs, should_train=False))
            for _ in range(n_tests)
        ]))

        print('\n==> epoch {}: sum_loss = {}'.format(epoch, loss))

        if loss < best_loss:
              best_loss = loss
              best_net = copy.deepcopy(optimizer.state_dict())
            
    return best_loss, best_net

class QuadraticLoss:
    '''quadratic function: f(theta)=|W*theta-y|_2^2.'''
    def __init__(self, **kwarg):
        self.W = torch.randn(10, 10)
        self.y = torch.randn(10)

    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y)**2)

class QuadOptimizee(nn.Module):
    def __init__(self, theta=None):
        super(QuadOptimizee, self).__init__()
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(10))
        else:
            self.theta = theta

    def forward(self, target):
        return target.get_loss(self.theta)

    def all_named_parameters(self):
        return [('theta', self.theta)]


class Optimizer(nn.Module):
    def __init__(self, preproc=False, D_H=20):
        super(Optimizer, self).__init__()
        self.D_H = D_H
        self.lstm0 = nn.LSTMCell(1, D_H)
        self.lstm1 = nn.LSTMCell(D_H, D_H)
        self.Linear = nn.Linear(D_H, 1)

    def forward(self, inputs, hidden, cell):
        h0, c0 = self.lstm0(inputs, (hidden[0], cell[0]))
        h1, c1 = self.lstm1(h0, (hidden[1], cell[1]))

        return self.Linear(h1), (h0, h1), (c0, c1)
 
def main():
    quadratic = QuadraticLoss()
    theta = torch.randn(10)
 #   print(quadratic.get_loss(theta))

    optimizee = QuadOptimizee()
#    print(optimizee(quadratic))
    loss, quad_optimizer = train_optimizer(QuadraticLoss, QuadOptimizee, lr=0.3)
    print('final loss:' , loss)
 
if __name__ == '__main__':
    main()
