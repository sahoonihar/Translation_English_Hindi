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
import sentencepiece as sp
import pandas as pd
from pathlib import Path
from indicnlp.tokenize import indic_tokenize

from model_pack.transformer import Transformer
from model_utility.translator import beam_search_bpe
from model_utility.utils import save_checkpoint

import nltk
from nltk.translate.bleu_score import SmoothingFunction
os.environ['CUDA_VISIBLE_DEVICES']='0'
cc = SmoothingFunction()

# Data Prep
# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")

en_sp = sp.SentencePieceProcessor()
en_sp.Load("./en_sp.model")

hi_sp = sp.SentencePieceProcessor()
hi_sp.Load("./hi_sp.model")

vocab_size = 60000

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


test_df = pd.read_csv('test_sm.csv')



# Evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (f"device used {device}.")


# Model hyper-parameters
src_vocab_size = len(english_txt.vocab)
trg_vocab_size = len(hindi_txt.vocab)
embedding_size = 256
num_heads = 8
num_layers = 1
dropout = 0.10
max_len = vocab_size
forward_expansion = 4
src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = 0

MODEL_PATH = f"./Model/model_bpe_{num_layers}_{vocab_size}_{embedding_size}.pth.tar"
output_file = Path(f"/home/nihar/Translation/model_bpe_{num_layers}_{vocab_size}_{embedding_size}.txt")
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
bleu_score_subword = []
with output_file.open('a', encoding='utf-8') as f:
    for idx in range(1000):
        english = test_df.eng_text.iloc[idx]
        actual_hindi = test_df.hindi_text.iloc[idx]
        try:
            #print(english)
            pred_word,pred_subword = beam_search_bpe(sentence=english, model=model, src_field=english_txt,
                                src_tokenizer=tokenize_eng, trg_field=hindi_txt, trg_vcb_sz=vocab_size,
                                k=10, max_ts=50, device=device)
            
            predicted_hindi = hi_sp.DecodePieces(pred_word)
            actual_subword = hi_sp.EncodeAsPieces(actual_hindi)
            actual_subword = " ".join([w for w in actual_subword])
            
            print (f"Test Set Index : {idx}")
            print (f"English Text : {english}")
            print (f"Original Hindi Text : {actual_hindi}")
            print (f"Predicted Hindi Text : {predicted_hindi}")
            print (f"Actual Hindi subword : {actual_subword}")
            print (f"Predicted Hindi subword : {pred_subword}")

            f.write(f"Test Set Index : {idx} \n")
            f.write(f"English Text : {english} \n")
            f.write(f"Original Hindi Text : {actual_hindi} \n")
            f.write(f"Predicted Hindi Text : {predicted_hindi} \n")
            f.write(f"Actual Hindi subword : {actual_subword} \n")
            f.write(f"Predicted Hindi subword : {pred_subword} \n")
            f.write("\n")
            
            references = actual_hindi.split(' ')
            hypothesis = predicted_hindi.split(' ')
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([references], hypothesis, weights = (1.0/4, 1.0/4, 1.0/4, 1.0/4), smoothing_function = cc.method2)
            bleu_scores.append(BLEUscore)

            
            
            references_subword = actual_subword.split(' ')
            hypothesis_subword = pred_subword.split(' ')
            BLEUscore_subword = nltk.translate.bleu_score.sentence_bleu([references_subword], hypothesis_subword, weights = (1.0/4, 1.0/4, 1.0/4, 1.0/4), smoothing_function = cc.method2)
            bleu_score_subword.append(BLEUscore_subword)
        except:
            continue
        
print(f"Average BLEU score : {sum(bleu_scores)/float(len(bleu_scores))}")
print(f"Average BLEU score at subword level : {sum(bleu_score_subword)/float(len(bleu_score_subword))}")
with output_file.open('a', encoding='utf-8') as f:
    f.write(f"Average BLEU score : {sum(bleu_scores)/float(len(bleu_scores))}")
    f.write(f"Average BLEU score at subword level : {sum(bleu_score_subword)/float(len(bleu_score_subword))}")
f.close()
    
