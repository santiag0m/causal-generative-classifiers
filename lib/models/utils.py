import torch


def multiply_probs_with_logits(logits_x, logits_y):
    """
    sum = x + y
    z = - log(exp(-sum) + exp(-x) + exp(-y))
    sigmoid(z) = sigmoid(x) * sigmoid(y)
    """
    logit_sum = logits_x + logits_y
    terms = torch.stack([logit_sum, logits_x, logits_y], dim=-1)
    terms = -1 * terms
    return -1 * torch.logsumexp(terms, dim=-1, keepdim=False)


def divide_probs_with_logits(logits_x, logits_y):
    """
    z = log(exp(0) + exp(-y)) - log(exp(-x) - exp(-y))
    sigmoid(z) = sigmoid(x) / sigmoid(y)
    """
    zeros = torch.zeros_like(logits_y)
    terms_left = torch.stack([zeros, -1 * logits_y])
    terms_left = torch.logsumexp(terms_left, dim=-1, keepdim=False)

    terms_right = -1 * torch.stack([logits_x, logits_y], dim=-1)
    # Do log sum exp trick
    c = torch.max(terms_right, dim=-1, keepdim=True).values
    terms_right = terms_right - c
    terms_right = c + torch.log(
        torch.exp(terms_right[..., 0]) - torch.exp([..., 1])
    )

    return terms_left - terms_right
