import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn
import torch

# Define the one_hot_flags function
def one_hot_flags(flag):
    # Example logic for one-hot encoding of flags
    flags_mapping = {'ACK': 0, 'PSH': 1, 'RST': 2, 'SYN': 3, 'FIN': 4}
    encoded = [0] * len(flags_mapping)
    for char in str(flag):
        if char in flags_mapping:
            encoded[flags_mapping[char]] = 1
    return encoded

# Load the dataset
df = pd.read_csv(
    '/home/binit/network_intrusion_detection_system/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv',
    low_memory=False
)

# Drop unnecessary columns
df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos', 'class', 'attackID', 'attackDescription'], errors='ignore')

# Replace invalid attackType values
df['attackType'] = df['attackType'].replace('---', 'bengin')
df['Date first seen'] = pd.to_datetime(df['Date first seen'], errors='coerce')  # Convert to datetime, handle invalid values

# lets count the labels with three most represented classes leaving other as two other are under 0.1%
count_labels = df['attackType'].value_counts() / len(df) * 100
plt.pie(count_labels[:3], labels=df['attackType'].unique()[:3], autopct='%.0f%%')

# plt.savefig("/home/binit/network_intrusion_detection_system/count_label.png")


fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(20, 5))
df['Duration'].hist(ax=ax1)
ax1.set_xlabel("Duration")
df['Packets'].hist(ax=ax2)
ax2.set_xlabel("Number of packets")
pd.to_numeric(df['Bytes'], errors='coerce').hist(ax=ax3)
ax3.set_xlabel("Number of bytes")
# plt.savefig("/home/binit/network_intrusion_detection_system/distribution.png")

# weekday
df['weekday'] = df['Date first seen'].dt.weekday
df = pd.get_dummies(df, columns=['weekday']).rename(columns={
    'weekday_0': 'Monday', 'weekday_1': 'Tuesday', 'weekday_2': 'Wednesday',
    'weekday_3': 'Thursday', 'weekday_4': 'Friday', 'weekday_5': 'Saturday',
    'weekday_6': 'Sunday',
})

# time stamp we can get time of the day
df['daytime'] = (df['Date first seen'].dt.second +
                 df['Date first seen'].dt.minute * 60 +
                 df['Date first seen'].dt.hour * 60 * 60) / (24 * 60 * 60)

# TCP flags
df = df.reset_index(drop=True)  # Reset index
ohe_flags = df['Flags'].apply(one_hot_flags).tolist()  # Apply one-hot encoding to Flags
df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohe_flags, columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])

# Source IP
temp_src = pd.DataFrame()
temp_src['SrcIP'] = df['Src IP Addr'].astype(str)
temp_src['SrcIP'][~temp_src['SrcIP'].str.contains(r'\d{1,3}\.', regex=True, na=False)] = '0.0.0.0'
temp_src = temp_src['SrcIP'].str.split('.', expand=True).rename(columns={2: 'ipsrc3', 3: 'ipsrc4'}).astype(int, errors='ignore')[['ipsrc3', 'ipsrc4']]
temp_src['ipsrc'] = temp_src['ipsrc3'].apply(lambda x: format(int(x), "b").zfill(8)) + temp_src['ipsrc4'].apply(lambda x: format(int(x), "b").zfill(8))
ipsrc_cols = [f'ipsrc_{i}' for i in range(16)]
temp_src_split = temp_src['ipsrc'].str.split('', expand=True).drop(columns=[0, 17]).rename(columns=dict(enumerate(ipsrc_cols)))
df = pd.concat([df, temp_src_split], axis=1)

# Destination IP
temp_dst = pd.DataFrame()
temp_dst['DstIP'] = df['Dst IP Addr'].astype(str)
temp_dst['DstIP'][~temp_dst['DstIP'].str.contains(r'\d{1,3}\.', regex=True, na=False)] = '0.0.0.0'
temp_dst = temp_dst['DstIP'].str.split('.', expand=True).rename(columns={2: 'ipdst3', 3: 'ipdst4'}).astype(int, errors='ignore')[['ipdst3', 'ipdst4']]
temp_dst['ipdst'] = temp_dst['ipdst3'].apply(lambda x: format(int(x), "b").zfill(8)) + temp_dst['ipdst4'].apply(lambda x: format(int(x), "b").zfill(8))
ipdst_cols = [f'ipdst_{i}' for i in range(16)]
temp_dst_split = temp_dst['ipdst'].str.split('', expand=True).drop(columns=[0, 17]).rename(columns=dict(enumerate(ipdst_cols)))
df = pd.concat([df, temp_dst_split], axis=1)

# Convert datetime columns to int64
for col in df.select_dtypes(include=['datetime64[ns]']).columns:
    df[col] = df[col].astype('int64')

# Handle missing or invalid values in the 'Bytes' column
m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull()].index
df.loc[m_index, 'Bytes'] = df.loc[m_index, 'Bytes'].apply(lambda x: 10e6 * float(str(x).strip().split()[0]))
df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

# One-hot encode 'Proto' and 'attackType' columns
labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']
df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Proto', 'attackType'])

# Ensure all label columns exist in the DataFrame
for label in labels:
    if label not in df.columns:
        df[label] = 0

# Split the dataset into train, validation, and test sets
df_train, df_test = train_test_split(df, random_state=0, test_size=0.2, stratify=df[labels])
df_val, df_test = train_test_split(df_test, random_state=0, test_size=0.5, stratify=df_test[labels])

# Apply scaling to numerical columns
scaler = PowerTransformer()
df_train[['Duration', 'Packets', 'Bytes']] = scaler.fit_transform(df_train[['Duration', 'Packets', 'Bytes']])
df_val[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_val[['Duration', 'Packets', 'Bytes']])
df_test[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_test[['Duration', 'Packets', 'Bytes']])

# Plot histograms for scaled data
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 5))
df_train['Duration'].hist(ax=ax1)
ax1.set_xlabel("Duration")
df_train['Packets'].hist(ax=ax2)
ax2.set_xlabel("Number of packets")
df_train['Bytes'].hist(ax=ax3)
ax3.set_xlabel("Number of bytes")
plt.savefig("/home/binit/network_intrusion_detection_system/distribution_for_scaled_data.png")
