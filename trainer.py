import torch
from tqdm import tqdm
from loss import info_nce_loss

class Trainer:
    def __init__(self, model, train_loader, optimizer, args):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.args = args

    def train(self):
        self.model.to(self.args.device)
        self.model.train()

        for epoch in range(self.args.epochs):
            total_loss = 0
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")

            for batch in loop:
                text1, text2 = batch['text1'], batch['text2']
                emb1 = self.model.encode(text1, convert_to_tensor=True, device=self.args.device)
                emb2 = self.model.encode(text2, convert_to_tensor=True, device=self.args.device)

                loss = info_nce_loss(emb1, emb2, self.args.temperature)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        self.model.save(self.args.save_dir)
