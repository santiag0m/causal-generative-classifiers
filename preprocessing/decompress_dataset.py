import os

from tqdm import tqdm
from torchvision.datasets import MNIST, CIFAR10

DATASETS = {
    "mnist": MNIST,
    "cifar10": CIFAR10,
}


def iterate_dataset(name: str, root: str, out_folder: str, train: bool):
    constructor = DATASETS[name]
    dataset = constructor(root, train=train, download=True)

    split = "train" if train else "test"
    split_folder = os.path.join(out_folder, split)

    os.makedirs(split_folder, exist_ok=True)

    for idx in tqdm(range(len(dataset))):
        image, label = dataset[idx]

        class_folder = os.path.join(split_folder, str(label))

        if not os.path.isdir(class_folder):
            os.makedirs(class_folder)

        image = image.convert("L")
        image.save(os.path.join(class_folder, f"{idx:05d}.jpg"))


def decompress_dataset(name: str, root: str = "./data"):
    out_name = f"class_{name}"
    out_folder = root.rstrip("/") + "/" + out_name
    iterate_dataset(name, root, out_folder, train=True)
    iterate_dataset(name, root, out_folder, train=False)
