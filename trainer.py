import torch
from tqdm import tqdm
from loss import MultiPositiveInfoNCE

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.args = args
        self.loss_fn = MultiPositiveInfoNCE(temperature=self.args.temperature)

    def train(self):
        self.model.to(self.args.device)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            total_train_loss = 0
            train_loop = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch+1}/{self.args.epochs}")

            for batch in train_loop:
                texts = batch['text']
                labels = batch['label']

                embeddings = self.model.encode(
                    texts, convert_to_tensor=True, device=self.args.device
                )

                loss = self.loss_fn(embeddings, labels.to(self.args.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                train_loop.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

            # ----------------------------
            # Validation
            # ----------------------------
            self.model.eval()
            total_val_loss = 0
            val_loop = tqdm(self.val_loader, desc=f"[Val] Epoch {epoch+1}/{self.args.epochs}")

            with torch.no_grad():
                for batch in val_loop:
                    texts = batch['text']
                    labels = batch['label']

                    embeddings = self.model.encode(
                        texts, convert_to_tensor=True, device=self.args.device
                    )

                    loss = self.loss_fn(embeddings, labels.to(self.args.device))
                    total_val_loss += loss.item()
                    val_loop.set_postfix(loss=loss.item())

            avg_val_loss = total_val_loss / len(self.val_loader)
            print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")

        # 모델 저장
        self.model.save(self.args.save_dir)

