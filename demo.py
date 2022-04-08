# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from pathlib import Path
import playsound
from playsound import playsound
from unidecode import unidecode
from gtts import gTTS
import wave
import os
import re
import cv2
import pyglet
import string
from time import sleep
pytesseract.pytesseract.tesseract_cmd = r'.\tesseract.exe'

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
import sentencepiece as sp
from indicnlp import common
import pandas as pd
from indicnlp.tokenize import indic_tokenize

from model_pack.transformer import Transformer
from model_utility.translator import beam_search
from model_utility.utils import save_checkpoint

import nltk
from nltk.translate.bleu_score import SmoothingFunction
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cc = SmoothingFunction()

'''#####################################################################'''

# Data Prep
# Settings for handling english text
spacy_eng = spacy.load("en_core_web_sm")

en_sp = sp.SentencePieceProcessor()
en_sp.Load("en_sp.model")

hi_sp = sp.SentencePieceProcessor()
hi_sp.Load("hi_sp.model")

vocab_size = 1000

# Defining Tokenizer
def tokenize_eng(text):
    return [tok.lower() for tok in en_sp.EncodeAsPieces(text)]


def tokenize_hindi(text):
    return [tok for tok in hi_sp.EncodeAsPieces(text)]

MODEL_PATH = "model_1_1000_64.pth.tar"

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
print (f"Eng vocab size : {src_vocab_size}, Hindi vocab size : {trg_vocab_size}")
embedding_size = 128
num_heads = 8
num_layers = 1
dropout = 0.10
max_len = vocab_size
forward_expansion = 4
src_pad_idx = english_txt.vocab.stoi["<pad>"]
trg_pad_idx = 0

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

'''###################################################################'''

## GUI code starts here.
root = Tk(  )
def readFimage():
    path = PathTextBox.get('1.0','end-1c')
    if path:
        im = Image.open(path) # the second one 
        im = im.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(2)
        im = im.convert('1')
        im.save('temp.jpg')
        text = pytesseract.image_to_string(Image.open('temp.jpg'))
        os.remove('temp.jpg')
        # im = Image.open(path)
        # text = pytesseract.image_to_string(im, lang = 'eng')
        ResultTextBox1.delete('1.0',END)
        ResultTextBox1.insert(END,text.encode(encoding = 'UTF-8'))
    else:
        ResultTextBox1.delete('1.0',END)
        ResultTextBox1.insert(END,"FILE CANNOT BE READ")
        
def translateHindi():
    english_sentence = ResultTextBox1.get('1.0', END+'-5c')
    predicted_hindi = beam_search(sentence=english_sentence, model=model, src_field=english_txt,
                            src_tokenizer=tokenize_eng, trg_field=hindi_txt, trg_vcb_sz=vocab_size,
                            k=5, max_ts=25, device=device)
    try:
        predicted_hindi = hi_sp.DecodePieces(predicted_hindi)
    except:
        predicted_hindi = "अनुवाद करने में असमर्थ"
        
    ResultTextBox2.delete('1.0',END)
    ResultTextBox2.insert(END,predicted_hindi)
    print (predicted_hindi)


def convertAudio():
    language = 'hi'
    tts = gTTS(text=ResultTextBox2.get('1.0','end-1c'), lang=language, slow=False)
    sound_file = 'hindi_audio.mp3'
    tts.save(sound_file)
    
    music = pyglet.media.load(sound_file, streaming=False)
    music.play()
    
    sleep(music.duration) #prevent from killing
    os.remove(sound_file) #remove temperory file

    

def OpenFile():
    name = askopenfilename(initialdir="./DLNLP/Project/",
                           filetypes =(("PNG File", "*.png"),("BMP File", "*.bmp"),("JPEG File", "*.jpeg")),
                           title = "Choose a file."
                           ) 
    PathTextBox.delete("1.0",END)
    PathTextBox.insert(END,name)
    
Title = root.title( "DLNLP Course Project.")
path = StringVar()

HeadLabel1 = Label(root,text="A complete pipeline for OCR-MT-TTS. It allows a user to\
listen to the contents of text images instead of reading through them. ")
HeadLabel1.grid(row = 1,column = 0,columnspan=4, rowspan=2)
# HeadLabel2 = Label(root,text=" Reader")
# HeadLabel2.grid(row = 1,column = 2)

InputLabel = Label(root,text = "INPUT IMAGE:")
InputLabel.grid(row=3,column = 1)

BrowseButton = Button(root,text="Browse",command = OpenFile)
BrowseButton.grid(row=3,column=2)

PathLabel = Label(root,text = "Path:")
PathLabel.grid(row = 4,column=1,sticky=(W))

PathTextBox = Text(root,height = 2)
PathTextBox.grid(row = 5,column = 1,columnspan=2)

ReadButton = Button(root,text="READ FROM IMAGE",command =readFimage)
ReadButton.grid(row = 6,column = 2)

DataLabel = Label(root,text = "DATA IN IMAGE:")
DataLabel.grid(row = 7,column=1,sticky=(W))

ResultTextBox1 = Text(root,height = 6)
ResultTextBox1.grid(row = 8,column = 1,columnspan=2)
# ResultTextBox1.pack()

ReadButton = Button(root,text="TRANSLATE TO HINDI",command =translateHindi)
ReadButton.grid(row = 9,column = 2)

DataLabel = Label(root,text = "HINDI TEXT:")
DataLabel.grid(row = 10,column=1,sticky=(W))

ResultTextBox2 = Text(root,height = 6)
ResultTextBox2.grid(row = 11,column = 1,columnspan=2)

ReadButton = Button(root,text="CONVERT TO AUDIO",command = convertAudio)
ReadButton.grid(row = 12,column = 2)





root.mainloop()