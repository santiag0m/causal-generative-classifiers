# Copied from: https://github.com/danielgreenfeld3/XIC/blob/master/hsic.py

import torch


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    # H = H.double().cuda()
    H = H.to(x.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC

def hsic_one_hot(residuals: torch.tensor, targets: torch.tensor) -> torch.Tensor:
    batch, num_classes, hidden_dim = residuals.shape
    # Select only class residuals for independence
    index = torch.reshape(targets, (batch, 1, 1))
    index = torch.broadcast_to(index, shape=(batch, num_classes, hidden_dim))
    residuals = torch.gather(residuals, dim=1, index=index)[:, 0, :]

    targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets = targets.to(residuals.device)

    # loss = 0
    # num_features = residuals.shape[-1]
    # for i in range(num_features):
    #     excluded = [j for j in range(num_features) if j != i]
    #     loss += HSIC(residuals[:, [i]], targets)
    #     loss += HSIC(residuals[:, [i]], residuals[:, excluded])

    loss = HSIC(residuals, targets)
    return loss
