import torch


def generate_uniform_weights(num_classes: int):
    return {str(i): 1.0 / num_classes for i in range(num_classes)}


def generate_random_weights(num_classes: int, temperature: float):
    weights = torch.softmax(torch.randn(num_classes) / temperature, 0)
    return {str(i): weights[i] for i in range(num_classes)}


def generate_class_unbalance(num_classes: int, one_to_ratio: int = 100):
    ratios = torch.tensor([1] + [one_to_ratio] * (num_classes - 1), dtype=torch.float32)
    weights = ratios / torch.sum(ratios)
    return {str(i): weights[i] for i in range(num_classes)}


def generate_single_class(num_classes: int):
    ratios = torch.tensor([1] + [0] * (num_classes - 1), dtype=torch.float32)
    weights = ratios / torch.sum(ratios)
    return {str(i): weights[i] for i in range(num_classes)}
