import torch
from torch.utils.data import Dataset

def generate_sequence(dataset, lookback=5, target='PM25_Concentration_0', future=1):
    X = dict()
    length = len(dataset)
    for i in range(length - lookback - future):
        seq = dataset.iloc[i:i + lookback,2:]
        seq = seq.values.tolist()
        des = dataset[target][i + lookback: i + lookback + future]
        des = des.values.tolist()
        X[i] = {"History":seq,"Target":des}
    return X

class SequenceDataset(Dataset):
    def __init__(self, df):
        self.data = df
        self.length = len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = torch.Tensor(sample['History'])
        y = torch.Tensor(sample['Target'])
        return x,y
    def __len__(self):
        return self.length
