import torch


def expectation_maximization(
    model,
    dataloader,
    y_marginal: torch.Tensor,
    hard: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    eps: float = 1e-6,
):
    if max_iterations < 1:
        raise ValueError("At least one (1) iteration is required")

    log_densities = get_feature_densities(model, dataloader)
    _, num_classes = log_densities.shape

    iters = 0
    while True:
        probs_y = torch.clamp(y_marginal, min=eps, max=1 - eps)  # (, Class)
        logits_y = torch.log(probs_y[None, ...])  # (1, Class)

        # Calculate posterior
        logits_joint = log_densities + logits_y
        logits_z = torch.log(torch.exp(logits_joint).sum(dim=1, keepdims=True))
        logits_y_z = logits_joint - logits_z

        if hard:
            # Take max likelihood prediction
            y_pred = torch.argmax(logits_y_z, dim=1, keepdims=False)
            y_pred = torch.nn.functional.one_hot(y_pred, num_classes=num_classes)
        else:
            # Take average prediction
            y_pred = torch.softmax(logits_y_z, dim=-1)

        # Calculate new marginal
        new_marginal = y_pred.sum(dim=0, keepdims=False)
        new_marginal = new_marginal / new_marginal.sum()

        diff = new_marginal - y_marginal
        diff = torch.max(torch.abs(diff))

        y_marginal = new_marginal
        iters += 1

        if (diff <= tolerance) or (iters >= max_iterations):
            return y_marginal


def get_feature_densities(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            preds = model.get_residual_densities(inputs)
            predictions.append(preds)
    predictions = torch.cat(predictions, dim=0)
    return predictions
