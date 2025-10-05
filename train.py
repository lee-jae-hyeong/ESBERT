from model import get_model
from data import get_dataloader
from trainer import Trainer
from train_arguments import TrainArguments
import torch

def train(train_csv: str):
    args = TrainArguments()
    args.display()

    model = get_model(args.model_name, args.pooling)
    dataloader = get_dataloader(train_csv, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(model, dataloader, optimizer, args)
    trainer.train()

if __name__ == "__main__":
    train("train.csv")
