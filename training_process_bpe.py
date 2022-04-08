# Basic packages
import os
import torch
import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

# Packages for data generator & preparation
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import sys
import sentencepiece as sp
from indicnlp import common
from indicnlp.tokenize import indic_tokenize

os.environ['CUDA_VISIBLE_DEVICES']='0'

# Packages for model building & inferences
from model_pack.transformer import Transformer
from model_utility.translator import beam_search
from model_utility.utils import save_checkpoint

# Data Prep
# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")

en_sp = sp.SentencePieceProcessor()
en_sp.Load("./en_sp.model")

hi_sp = sp.SentencePieceProcessor()
hi_sp.Load("./hi_sp.model")

vocab_size = 60000
MODEL_PATH = "./Model/"

# Defining Tokenizer
def tokenize_eng(text):
    return [tok.lower() for tok in en_sp.EncodeAsPieces(text)]


def tokenize_hindi(text):
    return [tok for tok in hi_sp.EncodeAsPieces(text)]


# Defining Field
english_txt = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")

# Defining Tabular Dataset
data_fields = [('eng_text', english_txt), ('hindi_text', hindi_txt)]
train_dt, val_dt = TabularDataset.splits(path='./', train='train_sm2.csv', validation='val_sm2.csv', format='csv', fields=data_fields)

# Building word vocab
english_txt.build_vocab(train_dt, max_size=vocab_size, min_freq=1)
hindi_txt.build_vocab(train_dt, max_size=vocab_size, min_freq=1)

# Training & Evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_model = True

# Training hyperparameters
num_epochs = 50
learning_rate = 3e-4
batch_size = 16

# Defining Iterator
train_iter = BucketIterator(train_dt, batch_size=batch_size, sort_key=lambda x: len(x.eng_text), shuffle=True)
val_iter = BucketIterator(val_dt, batch_size=batch_size, sort_key=lambda x: len(x.eng_text), shuffle=False)


# Model hyper-parameters
src_vocab_size = len(english_txt.vocab)
trg_vocab_size = len(hindi_txt.vocab)
print (f"English vocab size : {src_vocab_size}, Hindi vocab size : {trg_vocab_size}")
embedding_size = 256
num_heads = 8
num_layers = 1
dropout = 0.10
max_len = vocab_size
forward_expansion = 4
src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = 0

# Defining model & optimizer attributes
model2 = Transformer(src_vocab_size=src_vocab_size,
                    trg_vocab_size=trg_vocab_size,
                    src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    embed_size=embedding_size,
                    num_layers=num_layers,
                    forward_expansion=forward_expansion,
                    heads=num_heads,
                    dropout=dropout,
                    device=device,
                    max_len=max_len).to(device)

optimizer = optim.AdamW(model2.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

pad_idx = hindi_txt.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
loss_tracker = []
total_training_loss = []
total_val_loss =[]
early_stop_chk = 0
for epoch in range(num_epochs):
    model2.train()
    epoch_start_time =time.time()
    losses = []
    loop = tqdm(enumerate(train_iter), total=len(train_iter))
    for batch_idx, batch in loop:
        # Get input and targets and move to GPU if available
        # Switching axis because bucket-iterator gives output of size(seq_len,bs)
        inp_data = batch.eng_text.permute(-1, -2).to(device)
        target = batch.hindi_text.permute(-1, -2).to(device)

        # Forward prop
        output = model2(inp_data, target[:, :-1])

        optimizer.zero_grad()
        loss = criterion(output.reshape(-1, trg_vocab_size), target[:, 1:].reshape(-1))
        losses.append(loss.item())
        print(f'epoch {epoch}, batch_id {batch_idx}/{len(train_iter)}')

        # Checking GPU uses
        if device.type == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory/1024/1024
            allocated_mem = torch.cuda.memory_allocated(0)/1024/1024
            reserved_mem = torch.cuda.memory_reserved(0)/1024/1024
        else:
            total_mem = 0
            allocated_mem = 0
            reserved_mem = 0

        # Back prop
        loss.backward()

        # Clipping exploding gradients
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item(), total_gpu_mem=str(total_mem), gpu_allocated_mem=str(allocated_mem), gpu_reserved_mem=str(reserved_mem))

    train_mean_loss = sum(losses) / len(losses)
    scheduler.step(train_mean_loss)

    model2.eval()
    val_losses = []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for val_batch_idx, val_batch in tqdm(enumerate(val_iter), total=len(val_iter)):
            val_inp_data = val_batch.eng_text.permute(-1, -2).to(device)
            val_target = val_batch.hindi_text.permute(-1, -2).to(device)
            val_output = model2(val_inp_data, val_target[:, :-1])
            val_loss = criterion(val_output.reshape(-1, trg_vocab_size), val_target[:, 1:].reshape(-1))
            val_losses.append(val_loss.item())
        val_mean_loss = sum(val_losses)/len(val_losses)

    loss_tracker.append(val_mean_loss)
    total_training_loss.append(train_mean_loss)
    
    if save_model and val_mean_loss == np.min(loss_tracker):
        early_stop_chk = epoch
        checkpoint = {
            "state_dict": model2.state_dict(),
            "optimizer": optimizer.state_dict(),
  }
        save_checkpoint(MODEL_PATH, checkpoint, filename=f'model_bpe_{num_layers}_{vocab_size}_{embedding_size}.pth.tar')
    else:
        checkpoint = {
            "state_dict": model2.state_dict(),
            "optimizer": optimizer.state_dict(),
  }
        save_checkpoint(MODEL_PATH, checkpoint, filename=f'overfit_model_bpe_{num_layers}_{vocab_size}_{embedding_size}.pth.tar')
    
    if (epoch - early_stop_chk) >= 7:
        print(f"Epoch [{epoch + 1}/{num_epochs}]: train_loss= {train_mean_loss}; val_loss= {val_mean_loss},time taken for epoch: {time.time() - epoch_start_time}")
        break
    print(f"Epoch [{epoch + 1}/{num_epochs}]: train_loss= {train_mean_loss}; val_loss= {val_mean_loss},time taken for epoch: {time.time() - epoch_start_time}")
loss_dict = {'train':total_training_loss, 'val':loss_tracker}
pd.DataFrame(loss_dict).to_csv(f"./Loss/loss_bpe_{num_layers}_{vocab_size}_{embedding_size}.csv")
