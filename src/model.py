
"""
Author: Bhuvan Chennoju 
Date: 1st August 2024


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
        


