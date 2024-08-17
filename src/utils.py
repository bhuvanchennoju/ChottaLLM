import math

class LearningRateScheduler:

    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it+1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    

    def plot_lr(self, num_steps):
        import matplotlib.pyplot as plt
        lrs = [self.get_lr(it) for it in range(num_steps)]
        fig, ax = plt.subplots()
        ax.plot(lrs)
        ax.set(xlabel='step', ylabel='learning rate',
               title='Learning rate schedule')
        fig.savefig("lr_schedule.png")



from torch.distributed import init_process_group,destroy_process_group
import torch.distributed as dist
import torch

def setup_ddp(ddp_flag,ddp_config):
    if ddp_flag:
        if torch.cuda.is_available():
            assert torch.cuda.is_available(), "Distributed training requires CUDA"
            assert ddp_config['world_size'] > 1, "Distributed training requires at least 2 processes"
            assert ddp_config['rank'] >= 0, "Rank must be non-negative"
            assert ddp_config['rank'] < ddp_config['world_size'], "Rank must be smaller than world_size"

            init_process_group(backend='nccl')
            ddp_rank = ddp_config['rank']
            ddp_world_size = ddp_config['world_size']
            ddp_local_rank = ddp_config['local_rank']
            device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(device)

        else:
            print("CUDA is not available. Exiting.")
            exit()
    else:
        ddp_rank = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return ddp_rank, ddp_world_size, ddp_local_rank, device


def cleanup_ddp(ddp_flag):
    if ddp_flag:
        destroy_process_group()