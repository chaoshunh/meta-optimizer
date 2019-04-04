import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functools import reduce
from operator import mul

from torch.autograd import Variable
from layer_norm import LayerNorm1D
from layer_norm_lstm import LayerNormLSTMCell

from model import MLP
from utils import preprocess_gradients

class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model
        self.linear1 = nn.Linear(6, 2)
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []
        
        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))

        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)

        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads), flat_params.data, loss), 1))
        inputs = torch.cat((inputs, self.f, self.i), 1)

        self.f, self.i = self(inputs)

        # Meta update itself
        flat_params = self.f*flat_params - self.i*Variable(flat_grads)
        flat_params = flat_params.view(-1)

        
        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from thet meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model
        
        

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and
# applying updates to them.

class MetaModel:
    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.children():
            if isinstance(module, nn.modules.linear.Linear):
                module._parameters['weight'] = Variable(
                    module._parameters['weight'].data)
                module._parameters['bias'] = Variable(
                    module._parameters['bias'].data)
            else: # isinstance(module, nn.modules.container.Sequential
                for layer in module:
                    layer._parameters['weight'] = Variable(
                        layer._parameters['weight'].data)
                    layer._parameters['bias'] = Variable(
                        layer._parameters['bias'].data)

    def get_flat_params(self):
        params  = []
        for module in self.model.children():
            if isinstance(module, nn.modules.linear.Linear):
                params.append(module._parameters['weight'].view(-1))
                params.append(module._parameters['bias'].view(-1))
            else:
                for layer in module:
                    params.append(layer._parameters['weight'].view(-1))
                    params.append(layer._parameters['bias'].view(-1))

        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset  = 0
        
        for module in self.model.children():
            if isinstance(module, nn.modules.linear.Linear):
                weight_shape = module._parameters['weight'].size()
                bias_shape = module._parameters['bias'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters['weight'] = flat_params[
                    offset:offset+weight_flat_size].view(*weight_shape)
                module._parameters['bias'] = flat_params[
                    offset+weight_flat_size:offset+weight_flat_size+bias_flat_size].view(*bias_shape)
                offset += weight_flat_size+bias_flat_size
            else:
                for layer in module:
                    weight_shape = layer._parameters['weight'].size()
                    bias_shape = layer._parameters['bias'].size()

                    weight_flat_size = reduce(mul, weight_shape, 1)
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    layer._parameters['weight'] = flat_params[
                        offset:offset+weight_flat_size].view(*weight_shape)
                    layer._parameters['bias'] = flat_params[
                        offset+weight_flat_size:offset+weight_flat_size+bias_flat_size].view(*bias_shape)
                    offset += weight_flat_size+bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
    

def main():
    '''
    Create a meta optimizer that wraps a model into a meta model
    to keep track of the meta updates.
    '''
#    model = FullyConnectedNN()
    model = MLP()
    meta_model = MetaModel(model)
    meta_model.reset()
    flat_params = meta_model.get_flat_params()
    meta_model.set_flat_params(flat_params)

    x = torch.randn(1,1,28,28)
    meta_optimizer = MetaOptimizer(meta_model, num_layers=2, hidden_size=10)
    meta_optimizer.reset_lstm(model=model)

if __name__ == '__main__':
    main()
