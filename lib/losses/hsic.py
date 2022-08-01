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

def hsic_residuals(residuals: torch.tensor, targets: torch.tensor, featurewise: bool = True) -> torch.Tensor:
    batch, num_classes, num_feats = residuals.shape
    residuals = _index_residuals(residuals, targets)

    targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets = targets.to(residuals.device)

    if featurewise:
        loss = 0
        for i in range(num_feats):
            loss += HSIC(residuals[:, [i]], targets)
    else:
        loss = HSIC(residuals, targets)
    return loss

def hsic_prototypes(prototypes: torch.tensor, targets: torch.tensor, featurewise: bool = True) -> torch.Tensor:
    num_classes, num_feats = prototypes.shape
    prototypes = prototypes[targets, :]  # Batch x Feats

    targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets = targets.to(prototypes.device)

    if featurewise:
        loss = 0
        for i in range(num_feats):
            loss -= (1 / num_feats) * HSIC(prototypes[:, [i]], targets)
    else:
        loss -= HSIC(prototypes, targets)
    return loss

def hsic_features(features: torch.tensor, prototypes: torch.Tensor, targets: torch.tensor, featurewise: bool = True) -> torch.Tensor:
    batch, num_feats = features.shape
    num_classes, num_feats = prototypes.shape
    prototypes = prototypes[targets, :]  # Batch x Feats

    if featurewise:
        loss = 0
        for i in range(num_feats):
            loss -= (1 / num_feats) * HSIC(features[:, [i]], prototypes[:, [i]])
    else:
        loss -= HSIC(features, prototypes)
    return loss


def hsic_independence(residuals: torch.tensor, targets: torch.tensor, featurewise: bool = True) -> torch.Tensor:
    batch, num_classes, num_feats = residuals.shape
    residuals = _index_residuals(residuals, targets)

    targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    targets = targets.to(residuals.device)

    if featurewise:
        loss = 0
        for i in range(num_feats):
            excluded = [j for j in range(num_feats) if j != i]
            loss += HSIC(residuals[:, [i]], residuals[:, excluded])
    else:
        loss = HSIC(residuals, targets)
    return loss

def _index_residuals(residuals: torch.tensor, targets: torch.tensor) -> torch.Tensor:
    batch, num_classes, num_feats = residuals.shape
    
    # Select only class residuals for independence
    index = torch.reshape(targets, (batch, 1, 1))
    index = torch.broadcast_to(index, size=(batch, 1, num_feats))
    residuals = torch.gather(residuals, dim=1, index=index)[:, 0, :]
    return residuals
