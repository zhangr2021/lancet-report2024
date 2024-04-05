import pandas as pd
import numpy as np
import os
import re, string, timeit
import nltk
import sys
sys.path.append('/content/drive/MyDrive/hertie/lancet/lancet_2024')
from utils_lancet import *
from nltk.corpus import stopwords
from multiprocessing import Pool
import tqdm
import time
import re
import json
import multiprocessing
print("number of cpus: ", multiprocessing.cpu_count())

folder = "/content/drive/MyDrive/hertie/lancet/lancet_2024/result_2024/" #"English_2023/" #  "EU_result/" # 
folder_data = "/content/drive/MyDrive/hertie/lancet/lancet_2024/"
file = "multiglobal_english_text_2023.csv"

test = False

df = pd.read_csv(folder_data + file)

compoundlist = ["air pollution", "mental disorder", "mental disorders", "climate change", "changing climate", "climate emergency", 
          "climate crisis", "climate decay", "global warming", "green house", "extreme weather", "global environmental change", 
          "climate variability",  "low carbon", "renewable energy", "carbon emission", "carbon emissions", "carbon dioxide", 
          "co2 emission", "co2 emissions", "climate pollutant", "climate pollutants", "carbon neutral", "carbon neutrality", 
          "climate neutrality", "climate action", "net zero", "covid 19", "corona virus", "sars cov 2"]

health_dict =["malaria", "diarrhoea", "infection", "disease", "diseases", "sars", "measles", "pneumonia", "epidemic", 
                "epidemics", "pandemic", "pandemics", "epidemiology", "healthcare", "health", "mortality", "morbidity", 
                "nutrition", "illness", "illnesses", "ncd", "ncds", "air_pollution", "nutrition", "malnutrition", 
                "malnourishment", "mental_disorder", "mental_disorders", "stunting"]

climate_dict = ["climate_change", "changing_climate", "climate_emergency", "climate_crisis", "climate_decay", 
                 "global_warming", "green_house", "temperature", "extreme_weather", "global_environmental_change", 
                 "climate_variability", "greenhouse",  "greenhouse-gas", "low_carbon", "ghge", "ghges", "renewable_energy", 
                 "carbon_emission", "carbon_emissions", "carbon_dioxide", "carbon-dioxide", "co2_emission", "co2_emissions", 
                 "climate_pollutant", "climate_pollutants", "decarbonization", "decarbonisation", "carbon_neutral", 
                 "carbon-neutral", "carbon_neutrality", "climate_neutrality", "climate_action", "net-zero", "net_zero"]

def process_and_kwic(row):
    _, row = row
    tokens = text_process(row["text"])
    kwic_health = KWIC(search_list = health_dict, compound_list = compoundlist, corpus = tokens, id_ = row.Id)
    kwic_climate = KWIC(search_list = climate_dict, compound_list = compoundlist, corpus = tokens, id_ = row.Id)
    
    kwic_df_climate, _ = kwic_climate.kwic_search(window = 25,)
    kwic_df_health, tokens = kwic_health.kwic_search(window = 25,)
    
    return tokens, kwic_df_climate, kwic_df_health, row.Id

kwic_df_climate = pd.DataFrame()
kwic_df_health = pd.DataFrame()
merged_token = {"corpus": [], "Id": []}

cores = multiprocessing.cpu_count()
if test == True:
    cores = 24
    
if __name__ == '__main__':
    with Pool(cores) as p:
        if test == True:
            df = df.iloc[:24]
        r = list(tqdm.tqdm(p.imap(process_and_kwic, df.iterrows()), total = len(df)))
        for i in range(len(r)):
            corpus_, kwic_df_climate_, kwic_df_health_, id_ = r[i]
            merged_token["corpus"].append(corpus_)
            merged_token["Id"].append(id_)
            
            kwic_df_climate = pd.concat([kwic_df_climate, kwic_df_climate_])
            kwic_df_health = pd.concat([kwic_df_health, kwic_df_health_])
            
kwic_df_climate.to_pickle(folder + "df_kwic_climate.pkl")            
kwic_df_health.to_pickle(folder + "df_kwic_health.pkl")

#with open("EU_result/corpus_token.txt", "w") as fp:
 #   json.dump(merged_token, fp)

print('Process and Inidividual Kwic Done!')