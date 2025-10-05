from dataclasses import dataclass

@dataclass
class TrainArguments:
    model_name: str = "me5"
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    temperature: float = 0.07
    save_dir: str = "./saved_models/me5_finetuned"
    pooling: str = "mean"
    device: str = "cuda"

    def display(self):
        print("===== Training Arguments =====")
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("==============================")
