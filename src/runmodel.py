"""
Authored by: Bhuvan Chennoju
Created on: 4th August 2024

This is the main script to run the model training and evaluation.

"""

import os
import sys
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
import math


sys.path.append(str(Path.cwd()))
from src.dataUtils import CustomTokenizer, CustomDataset, CustomDataloader, DataLoaderLite
from src.utils import LearningRateScheduler
from src.model import GPTConfig, GPT

from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist




seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.set_float32_matmul_precision('high') # use float32 matmul precision for better performance

######################################### CONFIG  #########################################
WORK_dir = ''
DATA_dir =  os.path.join(WORK_dir, 'data')
SRC_dir = os.path.join(WORK_dir, 'src')
input_shards_path = os.path.join(DATA_dir,'wikitext2')
logs_dir = os.path.join(WORK_dir,'logs')
fig_dir = os.path.join(WORK_dir,'assets','images')

for dir in [logs_dir,fig_dir]:
    os.makedirs(dir,exist_ok=True)


total_batch_size = 524288 # 2^19 = 524288
B =  64
T = 1024

enc = CustomTokenizer() # tokenizer
#########################################  World setting   #########################################



ddp = int(os.environ.get('RANK',-1)) != -1 # check if the process is distributed
if ddp:
    assert torch.cuda.is_available(), "Distributed training requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # for logging and saving checkpoints

else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


print(f"ddp: {ddp}")
print(f"ddp_rank: {ddp_rank}")
print(f"ddp_world_size: {ddp_world_size}")
print(f"ddp_local_rank: {ddp_local_rank}")
print(f"master_process: {master_process}")


device_type = "cuda" if device.startswith('cuda') else "cpu"
print(f"device: {device}")
print(f"device_type: {device_type}")


######################################### Data loaders #########################################


assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size must be divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"grad_accum_steps: {grad_accum_steps}")
    print( f"total batch size: {total_batch_size}")

train_loader = DataLoaderLite(B = B, T = T, 
                              process_rank = ddp_rank, 
                              num_processes = ddp_world_size, 
                              split = 'train', 
                              master_process = master_process)

valid_loader = DataLoaderLite(B = B, T = T,
                              process_rank = ddp_rank,
                              num_processes = ddp_world_size,
                              split = 'valid',
                              master_process = master_process)


######################################### Model ################################################

cfg_dict = {
    'vocab_size': 50304, # vocab size for wikitext2 - gpt2 50304
    'block_size': 1024,
    'n_embed': 768,
    'n_layers': 1,
    'n_heads': 1,
    'dropout': 0.2,
    'biased': False,
    'causal': True,
    'compile': True
    
}

config = GPTConfig(cfg_dict)
model = GPT(config=config)
model.to(device)


if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])

if config.compile and device == "cuda":
    model = torch.compile(model)


raw_model = model.module if ddp else model

######################################### Optimizer and learning rate ################################################

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

lr_scheduler = LearningRateScheduler(max_lr, min_lr, warmup_steps, max_steps)
optimizer = raw_model.configure_optimizers(lr_rate=max_lr, wt_decay=0.1, betas=(0.9, 0.95), eps=1e-8, device_type = device_type)



######################################### Logging ################################################

log_dir = os.path.join(logs_dir, f"run_{ddp_rank}")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: 
    pass

########################################## TRAINING #########################################













# import time

# for step in range(max_steps):
#     t0 = time.time()
#     last_step = (step == max_steps - 1)

#     # once in a while evaluate our validation loss
#     if step % 250 == 0 or last_step:
#         model.eval()
#         valid_loader.reset()
#         with torch.no_grad():
#             val_loss_accum = 0.0
#             val_loss_steps = 20
#             for _ in range(val_loss_steps):
#                 x, y = valid_loader.next_batch()
#                 x, y = x.to(device), y.to(device)
#                 with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#                     logits, loss = model(x, y)
#                 loss = loss / val_loss_steps
#                 val_loss_accum += loss.detach()
#         if ddp:
#             dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
#         if master_process:
#             print(f"validation loss: {val_loss_accum.item():.4f}")
#             with open(log_file, "a") as f:
#                 f.write(f"{step} val {val_loss_accum.item():.4f}\n")
#             if step > 0 and (step % 5000 == 0 or last_step):
#                 # optionally write model checkpoints
#                 checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
#                 checkpoint = {
#                     'model': raw_model.state_dict(),
#                     'config': raw_model.config,
#                     'step': step,
#                     'val_loss': val_loss_accum.item()
#                 }
#                 # you might also want to add optimizer.state_dict() and
#                 # rng seeds etc., if you wanted to more exactly resume training
#                 torch.save(checkpoint, checkpoint_path)

#     # once in a while generate from the model (except step 0, which is noise)
#     if ((step > 0 and step % 250 == 0) or last_step) and (not config.compile):
#         model.eval()
#         num_return_sequences = 4
#         max_length = 32
#         tokens = enc.encode("Hello, I'm a language model,")
#         tokens = torch.tensor(tokens, dtype=torch.long)
#         tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#         xgen = tokens.to(device)
#         sample_rng = torch.Generator(device=device)
#         sample_rng.manual_seed(42 + ddp_rank)
#         while xgen.size(1) < max_length:
#             # forward the model to get the logits
#             with torch.no_grad():
#                 with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#                     logits, loss = model(xgen) # (B, T, vocab_size)
#                 # take the logits at the last position
#                 logits = logits[:, -1, :] # (B, vocab_size)
#                 # get the probabilities
#                 probs = F.softmax(logits, dim=-1)
#                 # do top-k sampling of 50 (huggingface pipeline default)
#                 # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#                 topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#                 # select a token from the top-k probabilities
#                 # note: multinomial does not demand the input to sum to 1
#                 ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
#                 # gather the corresponding indices
#                 xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#                 # append to the sequence
#                 xgen = torch.cat((xgen, xcol), dim=1)
#         # print the generated text
#         for i in range(num_return_sequences):
#             tokens = xgen[i, :max_length].tolist()
#             decoded = enc.decode(tokens)
#             print(f"rank {ddp_rank} sample {i}: {decoded}")

#     model.train()
#     optimizer.zero_grad()
#     loss_accum = 0.0
#     for micro_step in range(grad_accum_steps):
#         x, y = train_loader.next_batch()
#         x, y = x.to(device), y.to(device)
#         # added after video, this field is also used by the forward pass.
#         if ddp:
#             model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
#         # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
#         #     logits, loss = model(x, y)
            
#         logits, loss = model(x, y)
#         print(f"loss: {loss}")  
        
#         # we have to scale the loss to account for gradient accumulation,
#         # because the gradients just add on each successive backward().
#         # addition of gradients corresponds to a SUM in the objective, but
#         # instead of a SUM we want MEAN. Scale the loss here so it comes out right
#         loss = loss / grad_accum_steps
#         loss_accum += loss.detach()
#         loss.backward()
#     if ddp:
#         dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     # determine and set the learning rate for this iteration
#     lr = lr_scheduler.get_lr(step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     optimizer.step()
#     if device_type == "cuda":
#         torch.cuda.synchronize() # wait for the GPU to finish work
#     t1 = time.time()
#     dt = t1 - t0 # time difference in seconds
#     tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
#     tokens_per_sec = tokens_processed / dt
#     if master_process:
#         print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
#         with open(log_file, "a") as f:
#             f.write(f"{step} train {loss_accum.item():.6f}\n")

# if ddp:
#     destroy_process_group()























######################################## Data loaders #########################################

# data_dir =  input_shards_path
# trn_path, val_path, tst_path = data_dir + "/wikitext-2-raw-v1_train_000000.npy", data_dir + "/wikitext-2-raw-v1_validation_000000.npy", data_dir + "/wikitext-2-raw-v1_test_000000.npy"

# B = 64
# T = 1024

# process_rank = 0
# num_processes = 1
# master_process = True

# train_loader = DataLoaderLite(B,T,process_rank,num_processes,'train',master_process)
# valid_loader = DataLoaderLite(B,T,process_rank,num_processes,'valid',master_process)

# model = GPT(config)



# x,y = train_loader.next_batch()
# print(f"x shape: {x.shape}")
# print(f"y shape: {y.shape}")

# out, loss = model(x,y)
# print(loss)






# trn_dataset = CustomDataset(trn_path)
# val_dataset = CustomDataset(val_path)
# tst_dataset = CustomDataset(tst_path)


# print(f"train dataset length: {len(trn_dataset)}")
# print(f"valid dataset length: {len(val_dataset)}")
# print(f"test dataset length: {len(tst_dataset)}")

# trn_loader = CustomDataloader(trn_dataset, batch_size=32, shuffle=False)
# val_loader = CustomDataloader(val_dataset, batch_size=32, shuffle=False)
# tst_loader = CustomDataloader(tst_dataset, batch_size=32, shuffle=False)

# ######################################### Model #########################################

# model = GPT(config)

# x,y = next(iter(trn_loader))

# out, loss = model(x,y)

# print(f"output shape: {out.shape}")
# print(f"loss: {loss}")