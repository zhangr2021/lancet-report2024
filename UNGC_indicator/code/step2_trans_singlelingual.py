import os
import nltk
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from nltk import tokenize
import argparse

parser = argparse.ArgumentParser(description='csv_path')
parser.add_argument("--csv_path", type=str, default="")

# get it
args = parser.parse_args()
df_name = args.csv_path
folder = "/content/drive/MyDrive/hertie/lancet/lancet_2024/for_translation/"
os.environ['TORCH_HOME'] = os.getcwd()+'/cache'
lang = df_name.split("_")[1]

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=os.getcwd()+'/cache')
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=os.getcwd()+'/cache')
tokenizer.src_lang = lang

df_name = df_name
done_name = df_name.replace(".csv", "_d1.csv").replace("global", "result")

try:
    df = pd.read_csv(folder + df_name, lineterminator='\n')
except:
    df = pd.read_csv(folder + df_name)

done = ""
try:
    done = pd.read_csv(folder + done_name)
except:
    pass
    
#df = df[df.lang == lang].reset_index(drop = True)
print(lang, df_name, "original shape: ", df.shape)
df = df.iloc[len(done):].reset_index()
print(df.head())
print(lang, df_name, done_name, df.shape, "loaded!: ", len(done))

test = False 
if test == True:
    df = df.iloc[:40]

device = 'cuda'
batch = 16
print(device)
model= torch.nn.DataParallel(model)
model.to(device)
model.eval()

class Datapre(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = dataframe
        self.text = self.df.text
        self.max_len = max_len
        self.label = dataframe.Id.tolist()
        # check same length
        if (len(self.label) != len(self.text)):
            print('length does not match!')
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        label = self.label[index]
        
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            pad_to_max_length=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return  {
            'input_ids': torch.tensor(ids, dtype=torch.long).to(device),
            'attention_mask': torch.tensor(mask, dtype=torch.long).to(device),
        }, label
    

params = {'batch_size': batch,
          'shuffle': False,
          'num_workers': 0}

#prepare dataset
df_ = Datapre(df, tokenizer, max_len = 300)
loader_ = DataLoader(df_, **params)

listoftransids = []
listoftext = []
with torch.no_grad():
    for encoded, Ids in tqdm(loader_):
        listoftransids.extend(Ids)
        translated = model.module.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("en")).to(device)
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        listoftext.extend(translated_texts)
        if len(listoftransids) % 250 == 0:
            print('finished ',len(listoftransids))
            pd.DataFrame({'Id': listoftransids, 'text':listoftext}).to_csv(folder + df_name.replace(".csv", "").replace("global", "result") + '_tmp.csv', index= False)

pd.DataFrame({'Id': listoftransids, 'text':listoftext}).to_csv(folder + df_name.replace(".csv", "").replace("global", "result") + '.csv', index= False)
print('Translation finished!!!')