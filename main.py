import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn
import torch

df = pd.read_csv('/home/binit/network_intrusion_detection_system/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv')
df.head(5)

df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos','class', 'attackID', 'attackDescription'])


df['attackType'] = df['attackType'].replace('---', 'bengin')
df['Date first seen'] = pd.to_datetime(df['Date first seen'])

# lets count the labels with three most represented classes leaving other as two other are under 0.1%
count_labels = df['attackType'].value_counts() / len(df)* 100
plt.pie(count_labels[:3], labels=df['attackType'].unique()[:3], autopct='%.0f%%')

plt.savefig("/home/binit/network_intrusion_detection_system/count_label.png")


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3,
figsize=(20,5))
df['Duration'].hist(ax=ax1)
ax1.set_xlabel("Duration")
df['Packets'].hist(ax=ax2)
ax2.set_xlabel("Number of packets")
pd.to_numeric(df['Bytes'], errors='coerce').hist(ax=ax3)
ax3.set_xlabel("Number of bytes")
plt.savefig("/home/binit/network_intrusion_detection_system/distribution.png")
