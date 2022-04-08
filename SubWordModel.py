# -*- coding: utf-8 -*-

import sentencepiece as sp
import pandas as pd

train_df = pd.read_csv('train_sm2.csv')

def write_trainer_file(col, filename):
    texts = list(col.values)
    count = 0
    with open(filename, 'w',encoding='utf-8') as f:
        for text in texts:
            try:
                count += 1
                f.write(text + "\n")
            except:
                pass
            
        print(count)
en_trainer = "english.txt"
hi_trainer = "hindi.txt"

write_trainer_file(train_df["eng_text"], en_trainer)
write_trainer_file(train_df["hindi_text"], hi_trainer)

#create our English SentencePiece model
sp_en_train_param = f"--input={en_trainer} --model_prefix=en_sp --vocab_size=38428"
sp.SentencePieceTrainer.Train(sp_en_train_param)
en_sp = sp.SentencePieceProcessor()
en_sp.Load("en_sp.model")
 
#create our Dutch SentencePiece model
sp_hi_train_param = f"--input={hi_trainer} --model_prefix=hi_sp --vocab_size=46000"
sp.SentencePieceTrainer.Train(sp_hi_train_param)
hi_sp = sp.SentencePieceProcessor()
hi_sp.Load("hi_sp.model")

# print(en_sp.EncodeAsPieces("This is a test."))
# print(en_sp.EncodeAsIds("This is a test."))
# print(en_sp.DecodeIds(en_sp.EncodeAsIds("This is a test.")))
# print(en_sp.DecodePieces(en_sp.EncodeAsPieces("This is a test.")))

