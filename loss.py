import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 기존 1:1 InfoNCE
# ----------------------------
def info_nce_loss(embeddings_a, embeddings_b, temperature=0.07):
    embeddings_a = F.normalize(embeddings_a, dim=1)
    embeddings_b = F.normalize(embeddings_b, dim=1)

    logits = torch.matmul(embeddings_a, embeddings_b.T) / temperature
    labels = torch.arange(len(embeddings_a)).to(embeddings_a.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# ----------------------------
# Multi-Positive InfoNCE
# ----------------------------
def multi_positive_info_nce(embeddings, group_ids, temperature=0.07):
    """
    embeddings: (batch_size, hidden_dim) - anchor embeddings
    group_ids: (batch_size,) - 각 샘플이 속한 그룹 ID
    """
    embeddings = F.normalize(embeddings, dim=1)
    batch_size = embeddings.size(0)
    
    # similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # mask: 같은 그룹 = True, 자기 자신 제외
    group_ids = group_ids.unsqueeze(0)
    mask = group_ids == group_ids.T
    mask.fill_diagonal_(False)
    
    # log_softmax
    log_probs = F.log_softmax(sim_matrix, dim=1)
    
    # 각 anchor마다 positive 평균 log_prob 계산
    loss = 0
    for i in range(batch_size):
        pos_indices = mask[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) > 0:
            loss_i = -log_probs[i, pos_indices].mean()
            loss += loss_i
    loss = loss / batch_size
    return loss

class MultiPositiveInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, eps: float = 1e-10):
        """
        Multi-Positive InfoNCE Loss
        - features: (batch_size, hidden_dim)
        - group_ids: (batch_size,) 각 sample이 속한 그룹 ID
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        # L2 정규화
        features = F.normalize(features, dim=-1)
        batch_size = features.size(0)

        # similarity 계산 (batch 내 모든 sample과)
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Positive mask: 같은 그룹 & 자기 자신 제외
        group_ids = group_ids.unsqueeze(0)
        mask = (group_ids == group_ids.T).float()
        pos_mask = mask - torch.eye(batch_size, device=features.device)  # 자기 자신 제외
        neg_mask = 1 - mask  # negative mask

        # -1000으로 마스킹하여 logsumexp 시 안정화
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        # logsumexp 계산
        pos_exp_sum = (sim_matrix * pos_mask + pos_mask_add).logsumexp(dim=-1)
        all_exp_sum = (sim_matrix * pos_mask + pos_mask_add).logsumexp(dim=-1) + \
                      (sim_matrix * neg_mask + neg_mask_add).logsumexp(dim=-1)

        # loss 계산
        loss_per_example = torch.log((pos_exp_sum + self.eps) / (all_exp_sum + self.eps)).squeeze()
        loss = -loss_per_example.mean()
        return loss
