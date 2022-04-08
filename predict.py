# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# Basic packages
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Packages for data generator & preparation
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import sys
from indicnlp import common
import pandas as pd
from pathlib import Path
from indicnlp.tokenize import indic_tokenize

from model_pack.transformer import Transformer
from model_utility.translator import beam_search
from model_utility.utils import save_checkpoint

import nltk
from nltk.translate.bleu_score import SmoothingFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cc = SmoothingFunction()

# Data Prep
# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")


# Defining Tokenizer
def tokenize_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]


def tokenize_hindi(text):
    return [tok for tok in indic_tokenize.trivial_tokenize(text)]

vocab_size = 50000


# Defining Field
english_txt = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
hindi_txt = Field(tokenize=tokenize_hindi, init_token="<sos>", eos_token="<eos>")

# Defining Tabular Dataset
data_fields = [('eng_text', english_txt), ('hindi_text', hindi_txt)]
train_dt, val_dt = TabularDataset.splits(path='./', train='train_sm2.csv', validation='val_sm2.csv', format='csv', fields=data_fields)

# Building word vocab
english_txt.build_vocab(train_dt, max_size=vocab_size, min_freq=1)
hindi_txt.build_vocab(train_dt, max_size=vocab_size, min_freq=1)


test_df = pd.read_csv('test_sm.csv')



# Evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (f"device used {device}.")


# Model hyper-parameters
src_vocab_size = len(english_txt.vocab)
trg_vocab_size = len(hindi_txt.vocab)
print (f"English vocab size : {src_vocab_size}, Hindi vocab size : {trg_vocab_size}")
embedding_size = 512
num_heads = 8
num_layers = 2
dropout = 0.10
max_len = vocab_size
forward_expansion = 4
src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = 0

MODEL_PATH = f"./Model/model_{num_layers}_{vocab_size}_{embedding_size}.pth.tar"
output_file = Path(f"/home/nihar/Translation/model_{num_layers}_{vocab_size}_{embedding_size}_mod.txt")
if output_file.is_file():
    output_file.unlink()
else:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('a', encoding='utf-8') as f:
        f.write("Translation outputs : \n")

model = Transformer(src_vocab_size=src_vocab_size,
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

checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['state_dict'])

bleu_scores = []
with output_file.open('a', encoding='utf-8') as f:
    for idx in range(1000):
        english = test_df.eng_text.iloc[idx]
        actual_hindi = test_df.hindi_text.iloc[idx]
        try:
            predicted_hindi = beam_search(sentence=english, model=model, src_field=english_txt,
                                src_tokenizer=tokenize_eng, trg_field=hindi_txt, trg_vcb_sz=vocab_size,
                                k=10, max_ts=50, device=device)
            
            print (f"Test Set Index : {idx}")
            print (f"English Text : {english}")
            print (f"Original Hindi Text : {actual_hindi}")
            print (f"Predicted Hindi Text : {predicted_hindi}")
            f.write(f"Test Set Index : {idx} \n")
            f.write(f"English Text : {english} \n")
            f.write(f"Original Hindi Text : {actual_hindi} \n")
            f.write(f"Predicted Hindi Text : {predicted_hindi} \n")
            f.write("\n")
            
            references = actual_hindi.split(' ')
            hypothesis = predicted_hindi.split(' ')
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([references], hypothesis, weights = (1.0/4, 1.0/4, 1.0/4, 1.0/4), smoothing_function = cc.method2)
            bleu_scores.append(BLEUscore)
        except:
            continue
        
print(f"Average BLEU score : {sum(bleu_scores)/float(len(bleu_scores))}")
with output_file.open('a', encoding='utf-8') as f:
    f.write(f"Average BLEU score : {sum(bleu_scores)/float(len(bleu_scores))}")
f.close()
    
