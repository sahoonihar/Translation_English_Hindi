import pandas as pd
import re
from tqdm import tqdm
import copy
from model_utility.data_prep_utils import remove_sc, clean_text
from sklearn.model_selection import train_test_split


# Loading hindi text
with open("/home/nihar/OCR-MT-TTS/Dataset/parallel/IITB.en-hi.hi", "r", encoding='utf-8') as hindi_inp:
    _text = hindi_inp.read()
hindi_text = _text.split('\n')
print (len(hindi_text))



# Loading english text
with open("/home/nihar/OCR-MT-TTS/Dataset/parallel/IITB.en-hi.en", "r", encoding='utf-8') as eng_inp:
    _text = eng_inp.read()
eng_text = _text.split('\n')
print (len(eng_text))



# Removing Hindi sentences having english letter in it
ids_to_remove = {}
for _id, _t in tqdm(enumerate(hindi_text)):
    if len(re.findall(r'[a-zA-Z]', _t)) > 0:
        ids_to_remove[_id] = _t
    else:
        pass


ids_to_keep = [i for i in range(len(hindi_text)) if i not in ids_to_remove.keys()]
filtered_eng_text = []
filtered_hindi_text = []
for _id in tqdm(ids_to_keep):
    filtered_eng_text.append(eng_text[_id].lower())
    filtered_hindi_text.append(hindi_text[_id])


# Treating english sentences
clean_eng_text = []
for sent in tqdm(filtered_eng_text):
    clean_eng_text.append(clean_text(_text=copy.deepcopy(sent), lang="en"))


# Treating hindi sentences
clean_hindi_text = []
for sent in tqdm(filtered_hindi_text):
    clean_hindi_text.append(clean_text(_text=copy.deepcopy(sent), lang="hi"))


# Filtered Data
clean_data = pd.DataFrame({"eng_text": clean_eng_text, "hindi_text": clean_hindi_text})


# Filtering data based on sentence length
clean_data["eng_len"] = clean_data.eng_text.str.count(" ")
clean_data["hindi_len"] = clean_data.hindi_text.str.count(" ")
small_len_data = clean_data.query('eng_len < 50 & hindi_len < 50')


# Small set
small_data = small_len_data.loc[:, ["eng_text", "hindi_text"]].sample(n=300000)
train_set_sm, val_set_sm = train_test_split(small_data, test_size=0.3)
val_set, test_set = train_test_split(val_set_sm,test_size = 0.2)
train_set_sm.to_csv("train_sm2.csv", index=False)
val_set.to_csv("val_sm2.csv", index=False)
test_set.to_csv("test_sm2.csv", index=False)

