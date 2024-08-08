"""
Author: Bhuvan Chennoju
Date: 2nd Aug 2024

Description: All the required function to prepare the data for the model training.


"""


from tiktoken import get_encoding
import numpy as np
from torch import tensor,stack, int64,long
from torch.nn.utils.rnn import pad_sequence

class CustomTokenizer:
    def __init__(self,encoding = 'gpt2'):
        try:
            self.encoder = get_encoding(encoding) 
        except:
            raise Exception('Tokenizer not found, check the tokenizer name in tiktoken library')
        
        self.eot = self.encoder.eot_token
        
    def get_vocab_size(self):
        return self.encoder.n_vocab

    def encode(self,doc):
        tokens = [self.eot]
        tokens.extend(self.encoder.encode_ordinary(doc))
        return tokens
    
    def decode(self,tokens):
        return self.encoder.decode(tokens)
    


    def encode_np(self,doc):
        """
        This function encodes the document and returns the tokens in numpy array formatW
        """
        tokens = self.encode(doc)
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
    def decode_np(self,tokens):
        tokens = tokens.flatten()
        return self.decode(tokens)
    
    def encode_batch(self,docs):
        return [self.encode(doc) for doc in docs]
    
    def encode_batch_np(self,docs):
        return [self.encode_np(doc) for doc in docs]
    
    def decode_batch(self,tokens):
        return [self.decode(token) for token in tokens]
    
    def decode_batch_np(self,tokens):
        return [self.decode_np(token) for token in tokens]
    

class CustomDataset:
    def __init__(self,data_dir, block_size = 128):
        self.data_dir = data_dir
        self.block_size = block_size
        self.data = None
        self._load_data()

    def _load_data(self):
        self.data = np.load(self.data_dir)
        if self.data.dtype == np.int16:
            self.data = self.data.astype(np.int32)
        self.data = tensor(self.data,dtype = long) # converting to tensor
        return self.data
    
    def __len__(self):
        return len(self.data) // self.block_size 
    
    def __getitem__(self,idx):
        start_idx = idx * self.block_size
        end_idx = (start_idx + self.block_size) if (start_idx + self.block_size) < len(self.data) else len(self.data)
        inputs_ids = self.data[start_idx:end_idx]
        labels = self.data[start_idx+1:end_idx+1]

        if len(labels) < self.block_size:
            labels = np.append(labels,self.data[0:self.block_size-len(labels)])
        
        # inputs_ids = tensor(inputs_ids,dtype = int64)
        # labels = tensor(labels,dtype = int64).clone().detach()

        return inputs_ids,labels
    


class CustomDataloader:

    """
    purely memory based dataloader
    """

    def __init__(self,dataset:CustomDataset,batch_size = 64, shuffle = True, pad = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idxis = np.arange(len(self.dataset))
        self.pad = pad

        if self.shuffle:
            np.random.shuffle(self.idxis)
        self.str_idx = 0


    def __iter__(self):
        self.str_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxis)
        return self
    
    def __next__(self):
        if self.str_idx >= len(self.idxis):
            raise StopIteration
        
        end_idx = min(self.str_idx + self.batch_size,len(self.idxis)) #this is like the end case 
        batch_idxis = self.idxis[self.str_idx:end_idx]
        self.str_idx = end_idx

        batch = [self.dataset[i] for i in batch_idxis]
        inputs,labels = zip(*batch)
        inputs = stack(inputs)
        labels = stack(labels)

        if self.pad:
            inputs = pad_sequence(inputs,batch_first=True,padding_value = 0)
            labels = pad_sequence(labels,batch_first=True,padding_value = -100) # just to ensure the padding is not considered in the loss calculation

        return inputs,labels
    
    def __len__(self):
        return len(self.dataset) // self.batch_size


