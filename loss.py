import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelGroupInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, eps: float = 1e-10):
        """
        1:1 Label 기반 + Group ID 기반 Multi-Positive InfoNCE Loss
        - features: (batch_size, hidden_dim)
        - labels: (batch_size,) anchor와 대응 텍스트가 일치하는지 0/1
        - group_ids: (batch_size,) 동일 상품군 그룹 ID (없으면 -1)
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        # L2 정규화
        features = F.normalize(features, dim=-1)
        batch_size = features.size(0)

        # similarity 계산
        sim_matrix = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # ----------------------------
        # Positive mask 생성
        # ----------------------------
        # 1:1 label 기반
        labels = labels.unsqueeze(1)
        pos_mask_label = (labels == 1).float()  # 1:1 positive
        pos_mask_label.fill_diagonal_(0)       # 자기 자신 제외

        # group_id 기반
        group_ids = group_ids.unsqueeze(0)
        pos_mask_group = ((group_ids == group_ids.T) & (group_ids != -1)).float()
        pos_mask_group.fill_diagonal_(0)       # 자기 자신 제외

        # 최종 positive mask = 1:1 label positive OR group_id positive
        pos_mask = torch.clamp(pos_mask_label + pos_mask_group, 0, 1)
        neg_mask = 1 - pos_mask

        # ----------------------------
        # logsumexp 계산
        # ----------------------------
        pos_mask_add = neg_mask * (-1000)
        neg_mask_add = pos_mask * (-1000)

        pos_exp_sum = (sim_matrix * pos_mask + pos_mask_add).logsumexp(dim=-1)
        all_exp_sum = (sim_matrix * pos_mask + pos_mask_add).logsumexp(dim=-1) + \
                      (sim_matrix * neg_mask + neg_mask_add).logsumexp(dim=-1)

        # loss 계산
        loss_per_example = torch.log((pos_exp_sum + self.eps) / (all_exp_sum + self.eps)).squeeze()
        loss = -loss_per_example.mean()
        return loss

# ----------------------------
# 사용 예시
# ----------------------------
if __name__ == "__main__":
    batch_size = 6
    hidden_dim = 128

    # 더미 feature
    features = torch.randn(batch_size, hidden_dim)
    # label 1:1 매칭 (0/1)
    labels = torch.tensor([1,0,1,0,1,0])
    # group_id (동일 상품군, 없는 경우 -1)
    group_ids = torch.tensor([0,0,0,1,1,-1])

    loss_fn = LabelGroupInfoNCE()
    loss = loss_fn(features, labels, group_ids)
    print("Label+Group Multi-Positive InfoNCE Loss:", loss.item())
