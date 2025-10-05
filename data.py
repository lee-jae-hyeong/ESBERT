import pandas as pd
from torch.utils.data import Dataset, DataLoader

class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.text1 = df['anchor'].tolist()
        self.text2 = df['positive'].tolist()

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        return {
            'text1': self.text1[idx],
            'text2': self.text2[idx]
        }

def get_dataloader(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    dataset = ContrastiveDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
