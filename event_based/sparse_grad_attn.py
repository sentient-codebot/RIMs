
'''
Giving an N x M attention matrix, returns the same matrix,
but performs masking to determine where to block gradients.
'''

import numpy
import torch
from torch.autograd import Variable

from sparse_attn import Sparse_attention


class blocked_grad(torch.autograd.Function):
    """ 
    forward: make no change
    backward: block the gradients if mask[`] == 0
    """
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x # no change

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0 # gradient partially blocked with mask

class Sparse_grad_attention(torch.autograd.Function):
    ''' 
    - what does it really do?? 
    - same as blocked_grad, but with mask==sa(inp)
        i.e.    sga(inp, k) equivalent to 
                sa=Sparse_attention(top_k=k)
                mask=sa(inp)
                blocked_grad(inp, mask)
    '''

    @staticmethod
    def forward(ctx, inp, k):
        sa = Sparse_attention(top_k=k)
        sparsified = sa(inp) # is a mask
        ctx.save_for_backward(inp, sparsified)

        return inp # make no change

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float(), None 


if __name__ == "__main__":
    k = 2
    sga = Sparse_grad_attention.apply # autograd.Function
    sa = Sparse_attention(k) # nn.module

    x = torch.from_numpy(numpy.array([[[0.1, 0.0, 0.3, 0.2, 0.4],
                                       [0.5, 0.4, 0.1, 0.0, 0.0]]]))
    x = x.reshape((2, 5))

    x.requires_grad = True

    print(x)
    print('output', sga(x, k))

    (sga(x, k).sum()).backward()
    print('sparse grad', x.grad)

    # x = Variable(x.data, requires_grad=True)

    (sa(x).sum()).backward()

    print('normal grad', x.grad)
