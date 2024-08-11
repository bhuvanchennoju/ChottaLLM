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
from src.model import GPTConfig, GPT

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)

######################################### CONFIG  #########################################
WORK_dir = ''
DATA_dir =  os.path.join(WORK_dir, 'data')
SRC_dir = os.path.join(WORK_dir, 'src')
input_shards_path = os.path.join(DATA_dir,'wikitext2')
logs_dir = os.path.join(WORK_dir,'logs')
fig_dir = os.path.join(WORK_dir,'assets','images')

for dir in [logs_dir,fig_dir]:
    os.makedirs(dir,exist_ok=True)

# # config for the model
# cfg_dict = {
#     'vocab_size': 50204, # vocab size for wikitext2 - gpt2 50304
#     'block_size': 1024,
#     'n_embed': 768,
#     'n_layers': 12,
#     'n_heads': 12,
#     'dropout': 0.2,
#     'biased': False
# }
cfg_dict = {
    'vocab_size': 50304, # vocab size for wikitext2 - gpt2 50304
    'block_size': 1024,
    'n_embed': 768,
    'n_layers': 12,
    'n_heads': 12,
    'dropout': 0.2,
    'biased': False
}

config = GPTConfig(cfg_dict)



from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



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

    print(f"device: {device}")
    logging.info(f"device: {device}")


device_type = "cuda" if device.startswith('cuda') else "cpu"


torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)



enc = CustomTokenizer()
total_batch_size = 524288
B =  64
T = 1024

assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size must be divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"grad_accum_steps: {grad_accum_steps}")
    logging.info(f"grad_accum_steps: {grad_accum_steps}")

    print( f"total batch size: {total_batch_size}")
    logging.info(f"total batch size: {total_batch_size}")


train_loader = DataLoaderLite(B,T,ddp_rank,ddp_world_size,'train',master_process)
valid_loader = DataLoaderLite(B,T,ddp_rank,ddp_world_size,'valid',master_process)


torch.set_float32_matmul_precision('high')


model = GPT(config=config)
model.to(device)

use_compile = True

if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

print(model.get_num_params())


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizers(lr_rate=max_lr, wt_decay=0.1, betas=(0.9, 0.95), eps=1e-8, device_type = device_type)



log_dir = os.path.join(logs_dir,'gpt2')
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log_file = os.path.join(log_dir,'gpt2.log')
with open(log_file,'w') as f:
    f.write('')

logging.info(f"device: {device}")
logging.info(f"grad_accum_steps: {grad_accum_steps}")
logging.info(f"total batch size: {total_batch_size}")
logging.info(f"max_lr: {max_lr}")
logging.info(f"min_lr: {min_lr}")
logging.info(f"warmup_steps: {warmup_steps}")
logging.info(f"max_steps: {max_steps}")


######################################### TRAINING #########################################

import time

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        valid_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = valid_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

























######################################### Data loaders #########################################

# data_dir =  input_shards_path
# trn_path, val_path, tst_path = data_dir + "/wikitext-2-raw-v1_train_000000.npy", data_dir + "/wikitext-2-raw-v1_validation_000000.npy", data_dir + "/wikitext-2-raw-v1_test_000000.npy"

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