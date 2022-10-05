import torch


def expectation_maximization(model, dataloader, num_iterations=5, eps=1e-6):
    if num_iterations < 1:
        raise ValueError("At least one (1) iteration is required")

    log_densities = get_feature_densities(model, dataloader)
    _, num_classes = log_densities.shape

    # Assume uniform prior
    y_marginal = torch.ones((num_classes,), device=log_densities.device) / num_classes

    for _ in range(num_iterations):
        probs_y = torch.clamp(y_marginal, min=eps, max=1 - eps)  # (, Class)
        logits_y = torch.log(probs_y[None, ...])  # (1, Class)

        # Calculate posterior
        logits_joint = log_densities + logits_y
        logits_z = torch.log(torch.exp(logits_joint).sum(dim=1, keepdims=True))
        logits_y_z = logits_joint - logits_z

        # Take max likelihood prediction
        y_pred = torch.argmax(logits_y_z, dim=1, keepdims=False)
        y_pred = torch.nn.functional.one_hot(y_pred, num_classes=num_classes)

        # Calculate new marginal
        y_marginal = y_pred.sum(dim=0, keepdims=False)
        y_marginal = y_marginal / y_marginal.sum()

    return logits_y_z, y_marginal


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
