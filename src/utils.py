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

     