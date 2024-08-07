
"""
Author: Bhuvan Chennoju 
Date: 4th August 2024

kudos to:
    - Karpathy's: nanogpt repo 
    https://github.com/karpathy/nanoGPT/blob/master/model.py


"""

import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F



############################################################################################################
# for a transformer model I need following components:
# 1. MultiHeadAttention - for this autoregressive model I need to use causal self attention, like only decoder side of the transformer
# 2. FeedForward - a simple feedforward network with 2 linear layers and relu activation
# 3. TransformerBlock - a transformer block with multihead attention and feedforward network with layer norm and residual connections


@dataclass
class GPTConfig:
    def __init__(self,cfg):
        if cfg:
            self.vocab_size = cfg['vocab_size']
            self.block_size = cfg['block_size']
            self.n_embed = cfg['n_embed']
            self.n_heads = cfg['n_heads']
            self.n_layers = cfg['n_layers']
            self.dropout = cfg['dropout']
            self.biased = cfg['biased']
        else:
            self.vocab_size = 50304
            self.block_size = 64
            self.n_embed = 512
            self.n_heads = 8
            self.n_layers = 6
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
            w = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask = None, dropout_p = self.dropout if self.training else 0.0, is_causal = True)
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
    
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size,config.n_embed),
                wpe = nn.Embedding(config.block_size,config.n_embed),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
                ln_f = layer_norm(config.n_embed,config.biased),
            )
        )
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size, bias = config.biased) # this bisas has to be false
        # share weights between embedding and output layer
        self.transformer['wte'].weight = self.lm_head.weight
        self.apply(self._init_weights)

        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
            
        print(f"Number of parameters in the model: {self.get_num_params()/1e6}M")

    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())




    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
           std =0.02
           if hasattr(module, 'NANOGPT_SCALE_INIT'): # scale the initialization by the size of the model
                std *= (2 * self.config.n_layer) ** -0.5
           torch.nn.init.normal_(module.weight, mean=0, std=std)
           if module.bias is not None:
               torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
           torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    

    def forward(self,idx,targets = None):
        device = idx.device
        B,T = idx.size()
        # making sure input size is less than or equal to block size
        assert T <= self.config.block_size, f'input size is greater than context size {T} > {self.config.block_size}'

        # get the embeddings
        pos = torch.arange(0,T,device = device,dtype = torch.long) # (T,)

        #embeddings
        tok_emb = self.transformer['wte'](idx) # (B,T,C)
        pos_emb = self.transformer['wpe'](pos) # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        #transformer blocks
        for block in self.transformer['h']:
            x = block(x)

        #layer norm
        x = self.transformer['ln_f'](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index = -1)
            return logits,loss
        else:
            logits = self.lm_head(x[:,[-1],:])
            loss = None
        return logits,loss
    

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
    
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

    ## from here I am borrowing the code from the nanogpt repo

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

    def configure_optimizers(self,wt_decay,lr_rate,betas,device):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': wt_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")      
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu