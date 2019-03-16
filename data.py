import torch

def get_batch(batch_size):
    x = torch.randn(batch_size, 10)
    y = x.sum(1) # sum values of rows
    return x, y

if __name__ == '__main__':
    x, y = get_batch(2)
    print(x)
    print(y)
