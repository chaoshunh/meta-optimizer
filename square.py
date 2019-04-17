import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import copy
from torch.autograd import Variable

from logger import Logger

USE_CUDA = False

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

class SquareOptimizee(nn.Module):
    ''' Training f(theta) = theta^2.'''
    def __init__(self, theta=None):
        super().__init__()
        if theta is None:
            self.theta = nn.Parameter(torch.randn(1))
        else:
            self.theta = theta

    def forward(self):
        return self.theta**2

    def all_named_parameters(self):
        return [('theta', self.theta)]

class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
                
    def forward(self, inp, hidden, cell):
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def do_fit(opt_net, meta_opt, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    optimizee = w(target_to_opt())
    n_params = 0
    for p in optimizee.parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee()
                    
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data)
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()
            
        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()
                
            all_losses = None
                        
            optimizee = w(target_to_opt(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            
        else:
            optimizee = w(target_to_opt(**result_params))
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
            
    return all_losses_ever

def fit_optimizer(target_to_opt, preproc=False, unroll=2, optim_it=20, n_epochs=20, n_tests=100, lr=0.001, out_mul=1.0):
    opt_net = w(Optimizer(preproc=preproc))
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
    
    best_net = None
    best_loss = 100000000000000000

    logger = Logger('./logs')
    
    for epoch in (range(n_epochs)):
        for _ in range(10):
            do_fit(opt_net, meta_opt, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True)
        
        loss = (np.mean([
            np.sum(do_fit(opt_net, meta_opt, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=False))
            for _ in range(n_tests)
        ]))
        if (epoch+1) % 10 == 0:
            print('Epoch {} : Optimizer Loss: {:.4f}'.format(
                epoch+1, loss.item()))

                    #==============================================#
        #           Tensorboard Logging              #
        # ========================================== #

            # 1. Log scalar values (scalar summary)
            info = {'loss': loss.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in opt_net.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

        if loss < best_loss:
            #print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())

            
            
    return best_loss, best_net

def train(n_epochs=200, n_tests=100):
    optimizee = SquareOptimizee()

    result_params = {}
    logger = Logger('./logs')

    # Loss and optimizer
    for epoch in range(n_epochs):
        loss = optimizee()
        loss.backward()
        for name, param in optimizee.all_named_parameters():
            param_size = int(np.prod(param.size()))
            grads = param.grad.view(param_size, 1)
            
            updates = -grads * 0.01
            
            result_params[name] = param+updates.view(*param.size())*1.0
            result_params[name].retain_grad()

        optimizee = SquareOptimizee(**result_params)

        if (epoch+1) % 10 == 0:
            print('Epoch {} : Loss: {:.4f}'.format(
                epoch+1, loss.item()))

        #==============================================#
        #           Tensorboard Logging              #
        # ========================================== #
#            tensorboard_logging(loss, epoch, optimizee)

            # 1. Log scalar values (scalar summary)
            info = {'loss': loss.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in optimizee.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

def tensorboard_logging(loss, epoch, model):
    logger = Logger('./logs')
    # 1. Log scalar values (scalar summary)
    info = {'loss': loss.item()}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

    
if __name__ == '__main__':
    print('training...')
#    train()
    loss, quad_optimizer = fit_optimizer(SquareOptimizee,  lr=0.003, n_epochs=100)
    print(loss)
    
        
