import streamlit as st
import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='Syed Mushtaq Ali Trophy Performance Analysis Portal', layout='wide')
st.title('Syed Mushtaq Ali Trophy Performance Analysis Portal')
csv_files = [
    "Dataset/SMAT2.csv",
    "Dataset/SMAT1_part1.csv",
    "Dataset/SMAT1_part2.csv"
]
dataframes = [pd.read_csv(csv_file, low_memory=False) for csv_file in csv_files]
pdf = pd.concat(dataframes, ignore_index=True)
cols_conv=['season','match_id']
pdf[cols_conv] = pdf[cols_conv].astype(str)
bpdf=pdf
idf = pd.read_csv("Dataset/lifesaver_bat_smat.csv",low_memory=False)
info_df=pd.read_csv("Dataset/cricket_players_data.csv",low_memory=False)
bidf=pd.read_csv("Dataset/lifesaver_bowl_smat.csv",low_memory=False)
info_df=info_df.rename(columns={'player':'Player_name'})
pdf[['noballs', 'wides','byes','legbyes','penalty']] = pdf[['noballs', 'wides','byes','legbyes','penalty']].fillna(0).astype(int)
pdf['valid_ball'] = pdf.apply(lambda x: 1 if (x['wides'] == 0 and x['noballs'] == 0) else 0, axis=1)
