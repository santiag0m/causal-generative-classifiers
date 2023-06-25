from torchvision import transforms

HYPERPARAMS = {
    "mnist": {
        "epochs": 20,
        "learning_rate": 5e-2,
        "batch_size": 32,
        "weight_decay": 0,
        "momentum": 0,
        "num_layers": 1,
        "transform": transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    },
    "cifar10": {
        "epochs": 50,
        "learning_rate": 0.01,
        "batch_size": 128,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "num_layers": 2,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    },
    "fashion_mnist": {
        "epochs": 20,
        "learning_rate": 5e-2,
        "batch_size": 32,
        "weight_decay": 0,
        "momentum": 0,
        "num_layers": 1,
        "transform": transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    },
}
