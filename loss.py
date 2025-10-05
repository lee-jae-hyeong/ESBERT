import torch
import torch.nn.functional as F

def info_nce_loss(embeddings_a, embeddings_b, temperature=0.07):
    embeddings_a = F.normalize(embeddings_a, dim=1)
    embeddings_b = F.normalize(embeddings_b, dim=1)

    logits = torch.matmul(embeddings_a, embeddings_b.T) / temperature
    labels = torch.arange(len(embeddings_a)).to(embeddings_a.device)
    loss = F.cross_entropy(logits, labels)
    return loss
