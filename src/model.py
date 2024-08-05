
"""
Author: Bhuvan Chennoju 
Date: 4th August 2024

kudos to:
    - Karpathy's: nanogpt repo 
    https://github.com/karpathy/nanoGPT/blob/master/model.py


"""

import torch
import torch.nn as nn
import torch.nn.functional as F



############################################################################################################
# for a transformer model I need following components:
# 1. MultiHeadAttention - for this autoregressive model I need to use causal self attention, like only decoder side of the transformer
# 2. FeedForward - a simple feedforward network with 2 linear layers and relu activation
# 3. TransformerBlock - a transformer block with multihead attention and feedforward network with layer norm and residual connections


class Config:
    def __init__(self):
        self.n_embed = 512
        self.n_heads = 8
        self.block_size = 64
        self.dropout = 0.2
        self.biased = False



class CausalSelfAttention(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.block_size = config.block_size
        self.dropout = config.dropout
        self.biased = config.biased

        # multihead size
        assert config.n_embed % config.n_heads == 0, 'embedding dimension must be divisible by number of heads for splitting the heads in multihead attention'
        self.head_size = config.n_embed // config.n_heads

        # projections for q,k,v and output to vocab size
        self.qkv_proj = nn.Linear(self.n_embed, 3 * self.n_embed , bias = self.biased)
        self.out_proj = nn.Linear(self.n_embed,self.n_embed,bias = self.biased)
        
        # dropouts
        self.att_dropout = nn.Dropout(self.dropout)
        self.res_dropout = nn.Dropout(self.dropout)
        
        # causal mask
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') # check if the function is available
        if not self.flash:
            self.register_buffer('mask', torch.tril(torch.ones(self.block_size, self.block_size))).view(1, 1, self.block_size, self.block_size)

    def forward(self,x):
        B,T,C  = x.shape # batch size, context length, embedding dimension -- this is a input embedding after adding positional encoding to the input

        # project the input to q,k,v
        q,k,v = self.qkv_proj(x).split(self.n_embed, dim = 2)
        q = q.view(B,T,self.n_heads, self.head_size).transpose(1,2) # (B,T,C) --> (B,T,n_heads,head_size) --> (B,n_heads,T,head_size)
        k = k.view(B,T,self.n_heads, self.head_size).transpose(1,2) # (B,T,C) --> (B,T,n_heads,head_size) --> (B,n_heads,T,head_size)
        v = v.view(B,T,self.n_heads, self.head_size).transpose(1,2) # (B,T,C) --> (B,T,n_heads,head_size) --> (B,n_heads,T,head_size)


        # scaled dot product attention with causal mask
        # w = softmax(q @ k^T / sqrt(d_k)) @ v ;d_k = head_size
        # (B,T,n_heads,head_size) @ (B,n_heads,head_size,T) --> (B,n_heads,T,T)
        if self.flash:
            w = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask = None, dropout = self.dropout if self.training else 0.0, is_causal = True)
        else:
            w = (q @ k.transpose(-2,-1)) / (self.head_size ** 0.5)
            w = w.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
            w = F.softmax(w,dim = -1)
            w = self.att_dropout(w)
            w = w @ v # (B,n_heads,T,T) @ (B,n_heads,T,head_size) --> (B,n_heads,T,head_size)

        # combine the heads
        w = w.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        w = self.out_proj(w)
        w = self.res_dropout(w)
        return w


class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        self.ff = nn.Sequential(
            nn.Linear(self.n_embed,self.n_embed*4),
            nn.GELU(),
            nn.Linear(self.n_embed*4,self.n_embed)
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self,x):
        return self.dropout(self.ff(x))
    

class layer_norm(nn.Module):
    def __init__(self,n_embed,bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.bias = nn.Parameter(torch.zeros(n_embed)) if bias else None

    def forward(self,x):
        return F.layer_norm(x,self.weight.shape,self.weight,self.bias, 1e-5)


class TransformerBlock(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = layer_norm(config.n_embed,config.biased)
        self.norm2 = layer_norm(config.n_embed,config.biased)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
############################################################################################################