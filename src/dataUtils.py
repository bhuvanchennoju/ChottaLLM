"""
Author: Bhuvan Chennoju
Date: 2nd Aug 2024

Description: All the required function to prepare the data for the model training.


"""


import numpy as np
from tiktoken import get_encoding

class Tokenizer:
    
    def __init__(self,data):
        self.encoder = get_encoding("gpt2")

    def encode(self,doc):
        tokens = [self.encoder.eot]
        tokens.extend(self.encoder.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
    def decode(self,indices):
        return self.encoder.decode(indices)
    
    def write_datafile(self,filename,tokens_np):
        np.save(filename,tokens_np)

    def get_encoding(self):
        return self.encoder
    



class Dataloader:

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.idx = 0
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.idx >= len(self.indices):
            raise StopIteration
        
        batch_indices = self.indices[self.idx: self.idx + self.batch_size]
        self.idx += self.batch_size
        batch = [self.dataset[i] for i in batch_indices]
        
        tokens = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.dataset.tokenizer.c2i[self.dataset.tokenizer.pad_token])
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return tokens_padded, labels_padded
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


