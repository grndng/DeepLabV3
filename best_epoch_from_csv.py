import pandas as pd
import torch
from model import createDeepLabv3

#csv_path = r"D:\Potsdam_Final\Training_Results\Potsdam_512_H\log.csv"
#csv_path = "./Training_5ch/log.csv"
csv_path = r"C:\Users\dinga\Downloads\Potsdam_1024_rgbih_weights\log.csv"
df = pd.read_csv(csv_path)

print(df.head())
print(df.loc[df['Test_f1_score'].idxmax()])
