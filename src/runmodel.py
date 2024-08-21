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


from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils import setup_ddp


######################################### SEEDS  #########################################

seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.set_float32_matmul_precision('high') # use float32 matmul precision for better performance

######################################### CONFIG  #########################################
logging.basicConfig(level=logging.INFO)
logging.info(f"torch version: {torch.__version__}")


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


logging.info(f"total_batch_size: {total_batch_size}")
logging.info(f"B: {B}")
logging.info(f"T: {T}")

#########################################  World setting   #########################################



ddp = int(os.environ.get('RANK',-1)) != -1 
ddp_config = {
    'rank': int(os.environ.get('RANK',0)),
    'world_size': int(os.environ.get('WORLD_SIZE',1)),
    'local_rank': int(os.environ.get('LOCAL_RANK',0))
}

ddp_rank, ddp_world_size, ddp_local_rank, device = setup_ddp(ddp,ddp_config)

device_type = "cuda" if device.startswith("cuda") else "cpu"
master_process = ddp_rank == 0

logging.info(f"ddp: {ddp}")
logging.info(f"device: {device}")
logging.info(f"ddp_rank: {ddp_rank}")
logging.info(f"ddp_world_size: {ddp_world_size}")
logging.info(f"ddp_local_rank: {ddp_local_rank}")
logging.info(f"master_process: {master_process}")



######################################### Data loaders #########################################


assert total_batch_size % (B * T * ddp_world_size) == 0, "total batch size must be divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    logging.info(f"grad_accum_steps: {grad_accum_steps}")
    logging.info(f"total_batch_size: {total_batch_size}")

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

logging.info(f"config: {cfg_dict}")


config = GPTConfig(cfg_dict)
model = GPT(config=config)
model.to(device)


if ddp:
    model = DDP(model,device_ids=[ddp_local_rank])

if config.compile and device == "cuda":
    model = torch.compile(model)
    logging.info("model compiled")


raw_model = model.module if ddp else model

######################################### Optimizer and learning rate ################################################

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

lr_scheduler = LearningRateScheduler(max_lr, min_lr, warmup_steps, max_steps)
optimizer = raw_model.configure_optimizers(lr_rate=max_lr, wt_decay=0.1, betas=(0.9, 0.95), eps=1e-8, device_type = device_type)


########################################## TRAINING #########################################


from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import functional as F

logging.info("starting training")

log_dir = os.path.join(logs_dir, f"run_{ddp_rank}")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")


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

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not config.compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                logits = logits[:, -1, :] # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)
            
        logits, loss = model(x, y)
        print(f"loss: {loss}")  
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = lr_scheduler.get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()























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