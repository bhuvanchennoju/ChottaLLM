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


##########################################################################################


