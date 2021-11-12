import torch
import torch.nn as nn
import numpy

class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5):
        super(Sparse_attention,self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s, alpha=1, threshold=0.05):
        '''
        attn_s (BATCHSIZE, K) ??????
        or attn_s (2, K) ??????
        '''

        attn_plot = []
        eps = 10e-8
        # alpha = torch.clamp(alpha, min=0, max=1) # 

        # fixed_s = torch.zeros_like(attn_s)
        # fixed_s[:,:,0:-1] = attn_s[:,:,0:-1] * alpha.reshape(1,-1,1)
        # fixed_s[:,:,-1] = 1 - torch.sum(attn_s[:,:,0:-1]) * alpha.reshape(1,-1)

        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            #delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements 
            #delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            #delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
            delta = delta.reshape((delta.shape[0],1))


        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        #print('attn', attn_w_normalize)

        return attn_w_normalize


if __name__ == "__main__":
    k = 1
    print('take top k', k)
    sa = Sparse_attention(top_k=k)

    #batch x time

    x = torch.from_numpy(numpy.array([[[0.1, 0.0, 0.3, 0.2, 0.4],[0.5,0.4,0.1,0.0,0.0]]]))

    x = x.reshape((2,5))

    print('x shape', x.shape)
    print('x', x)

    o = sa(x)


    print('o', o)


