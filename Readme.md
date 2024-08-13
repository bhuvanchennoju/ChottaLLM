# ChottaGPT

This is project train a small scale LLM with wikipedia text and fine tune it for a specific task. 


## Table of Contents
- [ChottaGPT](#chottagpt)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Kernel Optimization with Tricks](#kernel-optimization-with-tricks)
    - [torch.compile](#torchcompile)
    - [Flash Attention](#flash-attention)
    - [Nice numbers](#nice-numbers)
  - [Algorithmic Optimization: Hyperparameters](#algorithmic-optimization-hyperparameters)
    - [AdamW optimizer](#adamw-optimizer)
    - [Gradient clipping](#gradient-clipping)
    - [Cosine Decay Learning Rate with Linear Warmup](#cosine-decay-learning-rate-with-linear-warmup)
    - [Linear Increase of Batch Size](#linear-increase-of-batch-size)
    - [Data Sampling Without Replacement](#data-sampling-without-replacement)
    - [Weight Decay and Fused Kernels in AdamW](#weight-decay-and-fused-kernels-in-adamw)

## Introduction

ChottaGPT is a lightweight implementation of GPT models inspired by NanoGPT, focusing on the key aspects of optimizing model training and inference. This README aims to document the various optimization techniques applied throughout the implementation, making it a valuable learning resource.


## Kernel Optimization with Tricks
### torch.compile
torch.compile performs kernel fusion, optimizing kernels by reducing Python overhead. Without torch.compile, operations would be performed multiple times on the chip and saved to GPU memory (HBM). With torch.compile, overheads like calculating activations (e.g., GELU) are performed only once when the tensors are on the GPU chip, and everything is stored to memory (HBM) in one go. This process is known as kernel fusion.

Speedup mainly comes from reducing Python overhead and GPU read/writes, and so the observed speedup may vary on factors such as model architecture and batch size. For example, if a model’s architecture is simple and the amount of data is large, then the bottleneck would be GPU compute and the observed speedup may be less significant.
   
Source: [PyTorch Torch Compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
While torch.compile is excellent for optimizing kernels, there are certain operations it might not optimize. An example of this is flash attention.

### Flash Attention
Flash Attention is a kernel fusion algorithm that fuses critical attention steps, including matrix multiplication, dropout, softmax, masking, and another matrix multiplication, into a single fused kernel called flash attention. The reason torch.compile cannot optimize it is that it requires an algorithmic rewrite in attention mechanism. From the paper, its reported that flash attention is 7.6 times faster than traditional attention because it is memory-conscious, despite having higher FLOPs than the traditional algorithm.

Under the hood, instead of performing a large chunk of matrix calculations in one step, the paper proposes an on-fly normalization calculation for softmax. This approach rewrites the softmax calculation into an incremental tiling fashion, making the physical existence of the large matrix multiplication redundant, performing the step on the fly. 
   ```python 
   # before 
   att = (q @ k.transpose(-2.-1)) * (1.0 / math.sqrt(k.size(-1))) #--->(1) 
   att = att.masked_fill(self.bias[:,:,:T,:T] == 0 , float('-inf'))
   att = F.softmax(att, dim = -1)
   y = att @ v # dims (B,nh, T, T) x (B, nh, T,hs) --> (B, nh, T, hs)

   # after
   y = F.scaled_dot_product_attention(q,k,v,is_causal = True)
   
   ```
Source: [Flash Attention GitHub Repository](https://github.com/Dao-AILab/flash-attention)


### Nice numbers
(or powers of two) Using powers of 2 is a hacky but effective way to optimize performance. In CUDA, many kernels use block tiles, which are typically in chunks of power of 2. When the desired calculation does not fit into these blocks, operations are performed in two or three phases, taking more time. By changing the numbers to nice powers of 2, we remove the boundary chunks that require a second phase of calculation, optimizing runtime.



## Algorithmic Optimization: Hyperparameters

This optimizations are purely copied based on the GPT-3 paper. I have explained breifly about each paprameter, and directly using the paramters discussed in the lecture.  

### AdamW optimizer 
The AdamW optimizer is an improved version of Stochastic Gradient Decent with momoentu, and known for its ability to handle sparse gradients on noisy problems. The choice of beta1 and beta2 values controls the decay rates of moving averages of gradient and its square, respectively, balancing the speed of convergence and stability. The eps value is a small constant added to avoid division by zero errors.

- **beta1 = 0.9**
- **beta2 = 0.95**
- **eps = 10^-8**

### Gradient clipping
 Clipping the gradients to 1 after the backward pass to prevent traing overshoots, and from updating with excessively large gradients, which can destabilize or shock the model in the training process. High gradient values, resulting from high loss, can cause the model to take large, destabilizing steps during optimization. For instance, one of the batch have bad quality data, that would get the high loss and sudden spike in the gradient calculation, and this shock the process. By clipping the gradients, we ensure that the model updates remain within a reasonable range, promoting stable and consistent learning.
1) 
```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### Cosine Decay Learning Rate with Linear Warmup
Adopting a cosine decay learning rate schedule helps in smoothly reducing the learning rate, preventing abrupt changes that can destabilize training. The initial linear warmup phase allows the model to gradually adapt to learning, reducing the risk of overshooting the optimal solution during the early stages of training when the weights are randomly initialized and gradients can be large.

```python

# cosine decay learning rate by karpathy
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(step):
  # 1) linear warmup for warmup_steps steps
  if step < warmup_steps:
      return max_lr * (step + 1) / warmup_steps
  
  # 2) if step > max_steps, return min learning rate
  if step > max_steps:
      return min_lr

  # 3) in between, use cosine decay down to min learning rate
  decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)


# in training loop
for step in range(max_steps):
  # code 
  # code
  # backward, and norm
  lr = get_lr(step) #<----------- learning rate is changing in every step
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr  
  optimizer.step()

```


### Linear Increase of Batch Size
Gradually increasing the batch size can lead to more stable convergence. Starting with smaller batch sizes helps the model learn more nuanced patterns in the data, while progressively larger batches improve computational efficiency and make better use of GPU resources. This is not implemented in the lecture series yet. 

### Data Sampling Without Replacement
Sampling data without replacement ensures that each sample is seen exactly once per epoch. This approach minimizes overfitting by preventing the model from seeing the same samples too frequently, which can cause it to memorize rather than generalize from the training data. In the code, this is implemented by step wise fashioned traversing the dataset for given epochs. 

### Weight Decay and Fused Kernels in AdamW
In the Adamw, weight decay acts as a regularizer, preventing the model from becoming overly complex and reducing the risk of overfitting. In the code implementation a weight decay of 0.1 is set. From the lecture, its is stated that weight decay should be applied to only tensors that are not 1 dim.  By splitting parameters into those that need weight decay and those that don't (e.g., layer norms, scales, biases), and using fused AdamW, we optimize the training process. Fused AdamW reduces the overhead of the optimization process by merging multiple operations into a single kernel call, enhancing computational efficiency on CUDA-enabled GPUs.
