import torch.nn.functional as F
from torch import nn
from math import sqrt
import torch


class Encoder(nn.Module):
    """
        A Key & Query are contracted then normalized via softmax,
        scaling the determinant to somewhere within [0,1].
        Then contraction with the Value.
    """
    def __init__(self, d_k, d_v, d_a):
        super().__init__()

        self.encoders = nn.ModuleList([nn.Parameter(torch.randn(d_a,d)) for d in [d_k, d_k, d_v]])

    def forward(self, query, key, value):
        [Q, K, V] = [torch.matmult(E, M) for E, M in zip(self.encoders, [query, key, value])]
        return Q, K, V


class KeyValueMask(nn.Module):
    """
        A Query/Key dot product is scaled before softmax normalization,
        the result is operated on the Value.
    """
    def __init__(self, d_k, attn_dropout=0):
        super().__init__()
        self.scaling = 1/d_k
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V):
        dots = torch.matmul(Q, K.T()).mul(self.scaling)  # Operates a Key on each Query and scales
        mask = self.dropout(F.softmax(dots, dim=-1))
        output = torch.matmul(mask, V)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, heads=3, d_a=None):
        super().__init__()
        if not d_a:
            d_a = heads*d_v

        self.encoders = nn.ModuleList([Encoder(d_k, d_v, d_a) for _ in range(heads)])
        self.ApplyAttention = KeyValueMask(d_k)
        self.Reshape = nn.Parameter(nn.randn(self.heads*d_v, d_a))

    def forward(self, queries, keys, values):
        A = torch.concat([self.ApplyAttention(encoder(queries, keys, values)) for encoder in self.encoders], dim=-1)
        return torch.matmul(A, self.Reshape)


class FeedForward(nn.Module):
    def __init__(self, d_a):
        MyLinear(4, 3),
        nn.ReLU(),
        MyLinear(3, 1)

    def __init__(self, d_a, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_a, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_a) # position-wise
        self.layer_norm = nn.LayerNorm(d_a, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output += residual

        return output

sample_input = torch.randn(4)
net(sample_input)

# Using Sequential to create a small model. When `model` is run,
# input will first be passed to `Conv2d(1,20,5)`. The output of
# `Conv2d(1,20,5)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Conv2d(20,64,5)`. Finally, the output of
# `Conv2d(20,64,5)` will be used as input to the second `ReLU`
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

x = torch.randn(5, requires_grad=True)
y = x.pow(2)
