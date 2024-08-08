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


sys.path.append(str(Path.cwd()))
from src.dataUtils import CustomTokenizer, CustomDataset, CustomDataloader
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

# config for the model
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


######################################### Data loaders #########################################

data_dir =  input_shards_path
trn_path, val_path, tst_path = data_dir + "/wikitext-2-raw-v1_train_000000.npy", data_dir + "/wikitext-2-raw-v1_validation_000000.npy", data_dir + "/wikitext-2-raw-v1_test_000000.npy"

trn_dataset = CustomDataset(trn_path)
val_dataset = CustomDataset(val_path)
tst_dataset = CustomDataset(tst_path)


print(f"train dataset length: {len(trn_dataset)}")
print(f"valid dataset length: {len(val_dataset)}")
print(f"test dataset length: {len(tst_dataset)}")

trn_loader = CustomDataloader(trn_dataset, batch_size=32, shuffle=False)
val_loader = CustomDataloader(val_dataset, batch_size=32, shuffle=False)
tst_loader = CustomDataloader(tst_dataset, batch_size=32, shuffle=False)

######################################### Model #########################################

model = GPT(config)

x,y = next(iter(trn_loader))

out, loss = model(x,y)

print(f"output shape: {out.shape}")
print(f"loss: {loss}")