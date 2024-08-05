"""
Authored by: Bhuvan Chennoju
Created on: 21st July 2024

Kudos to:
    - Karpathy's: https://github.com/karpathy/build-nanogpt?tab=readme-ov-file
    - Video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=784s

this is a simple train, and validation loop for the bigram model.

"""
import torch
import logging

def train(model,optimizer,epochs,batcher,eval_iters):

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train','valid']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x,y = batcher.get_batch(split)
                _, loss = model(x,y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

        model.train()
        return out

    loss_track = {'train':[],'valid':[]}
    for epoch in range(epochs):

        if epoch % eval_iters == 0:
            losses = estimate_loss()
            loss_track['train'].append(losses['train'])
            loss_track['valid'].append(losses['valid'])
            logging.info(f'epoch:{epoch}, train_loss:{losses["train"]}, valid_loss:{losses["valid"]}')
            print(f'epoch:{epoch}, train_loss:{losses["train"]}, valid_loss:{losses["valid"]}')

        x,y = batcher.get_batch('train')
        _, loss = model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()    
    return loss_track

