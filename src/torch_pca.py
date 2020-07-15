import torch

a = torch.tensor([[2,3,4,5,6,7,8,9],[1,3,2,7,6,4,4,3],[6,2,3,4,1,2,4,4]])

b = torch.pca_lowrank(a,q=3)

print(b)