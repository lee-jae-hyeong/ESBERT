import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ContrastiveDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()           # 모든 텍스트
        self.group_ids = df['group_id'].tolist()   # 각 sample의 그룹/라벨

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'group_id': self.group_ids[idx]
        }

def get_dataloaders(csv_path, batch_size=32, val_ratio=0.1, random_state=42):
    df = pd.read_csv(csv_path)

    # train / validation split
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=random_state)

    # Dataset & DataLoader 생성
    train_dataset = ContrastiveDataset(train_df)
    val_dataset = ContrastiveDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# ----------------------------
# 사용 예시
# ----------------------------
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders("contrastive_data.csv", batch_size=16, val_ratio=0.2)
    for batch in train_loader:
        print(batch['text'])
        print(batch['group_id'])
        break

