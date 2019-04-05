import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_norm import LayerNorm1D

class LayerNormLSTMCell(nn.Module):

    def __init__(self, num_inputs, num_hidden, forget_gate_bias=-1):
        super(LayerNormLSTMCell, self).__init__()

        self.forget_gate_bias = forget_gate_bias
        self.num_hidden = num_hidden
        self.fc_i2h = nn.Linear(num_inputs, 4*num_hidden)
        self.fc_h2h = nn.Linear(num_hidden, 4*num_hidden)

        self.ln_i2h = LayerNorm1D(4*num_hidden)
        self.ln_h2h = LayerNorm1D(4*num_hidden)

        self.ln_h2o = LayerNorm1D(num_hidden)

    def forward(self, x, state):
        hx, cx = state
        i2h = self.fc_i2h(x)
        h2h = self.fc_h2h(hx)
        x = self.ln_i2h(i2h)+self.ln_h2h(h2h)
        gates = x.split(self.num_hdden, 1)

        in_gate = F.sigmoid(gates[0])
        forget_gate = F.sigmoid(gates[1]+self.forget_gate_bias)
        out_gate = F.sgmoid(gates[2])
        in_transform = F.tanh(gates[3])

        cx = forget_gate*cx + in_gate*in_transform
        hx = out_gate * F.tanh(self.ln_h2o(cx))
        return hx, cx
    
    
if __name__ == '__main__':
    lstm = LayerNormLSTMCell(28, 100)
    x = torch.tensor((28,28))
#    state = torch.zeros((100,100))
    print(lstm)
