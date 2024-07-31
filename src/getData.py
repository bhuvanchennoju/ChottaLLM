"""
Author: Bhuvan Chennoju
Date: 27th July 2024
Description: This script downloads the wikitext2 data to disk. "mindchain/wikitext2"
"""

import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from tiktoken import get_encoding
from multiprocessing import Pool, cpu_count


class Wikitext2:
    def __init__(self, local_dir, remote_name, shard_size, tokenizer_name="gpt2"):
        self.local_dir = local_dir
        self.remote_name = remote_name
        self.shard_size = shard_size
        self.DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), self.local_dir)
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)

        self.encoder = get_encoding(tokenizer_name)
        self.eot = self.encoder._special_tokens['<|endoftext|>']

    def download(self):
        wikitext = load_dataset("wikitext", name=self.remote_name)
        return wikitext

    def tokenize(self, doc):
        tokens = [self.eot]
        tokens.extend(self.encoder.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def download_data(self):
        wikitext = self.download()

        nprocs = max(1, cpu_count() // 2)
        with Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for tokens in pool.imap(self.tokenize, wikitext, chunksize=16):
                if token_count + len(tokens) < self.shard_size:
                    all_tokens_np[token_count:token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(self.DATA_CACHE_DIR, f"wikitext_{split}_{shard_index:06d}")
                    remainder = self.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                    self.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder
            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(self.DATA_CACHE_DIR, f"wikitext_{split}_{shard_index:06d}")
                self.write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    local_dir = "/n/projects/kc2819/projects/ChotaLLM/data"
    remote_name = "wikitext-2-raw-v1"
    shard_size = int(1e8)

    wikitext2_instance = Wikitext2(local_dir, remote_name, shard_size)
    wikitext2_instance.download_data()
