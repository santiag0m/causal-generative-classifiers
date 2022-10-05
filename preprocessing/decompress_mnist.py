import os

from tqdm import tqdm
from torchvision.datasets import mnist


def iterate_mnist(root: str, out_folder: str, train: bool):
    dataset = mnist.MNIST(root, train=train, download=True)

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


def main(root: str = "./data", out_folder: str = "./data/class_mnist"):
    iterate_mnist(root, out_folder, train=True)
    iterate_mnist(root, out_folder, train=False)


if __name__ == "__main__":
    main()
