
"""
Author: Bhuvan Chennoju 
Date: 1st August 2024

kudos to:
    - Karpathy's: nanogpt repo 
    https://github.com/karpathy/nanoGPT/blob/master/model.py

"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # to make sure n_embd is divisible by n_head so that the output can be concatenated
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # 3 times the n_embd because we need q,k,v
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # to project the output of the attention to n_embd
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop) # dropout for attention
        self.resid_dropout = nn.Dropout(config.resid_pdrop) # dropout for residual connection
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop

        # flash attention parameters
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if self.flash:
            self.flash_attention = torch.nn.functional.scaled_dot_product_attention
        else:
            print('Using standard attention, as flash attention is only available in PyTorch >= 2.0.0')
            print(torch.__version__)
            ## register buffer that will be used for the attention mask
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
            
    
    def forward(self,x):
        B,T,C = x.size()  # B: batch size, T: sequence length, C: n_embd

        q,k,v = self.c_attn(x).split(self.n_embd, dim = 2) # split the output of the linear layer into q,k,v
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # reshape q to (B,n_head,T,C//n_head)
        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2)

        # flash attention (B,nh T, C//n_head) X (B,nh, C//n_head, T) -> (B,nh,T,T)

        if self.flash:
            y = self.flash_attention(q, k, v, attn_mask = None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B,n_head,T,T) X (B,n_head,T,C//n_head) -> (B,n_head,T,C//n_head)

        y = y.transpose(1,2).contiguous().view(B,T,C) # reshape y to (B,T,C)

        y = self.resid_dropout(self.c_proj(y)) # apply residual dropout and project the output of attention to n_embd
        return y
    


    class LayerNorm(nn.)