from random import Random
from typing import List, Tuple, Dict, Optional, Callable

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes, make_dataset


class ImbalancedImageFolder(ImageFolder):
    def __init__(
        self, root: str, class_weights: Dict[str, float], seed: int = 0, *args, **kwargs
    ):

        if sum(class_weights.values()) != 1:
            raise ValueError("Class weights do not sum to 1")

        for v in class_weights.values():
            if v < 0:
                raise ValueError("Negative class weights are not allowed")

        self.class_weights = class_weights
        self.rng = Random(seed)

        # Get classes first
        self.classes, self.class_to_idx = self.find_classes(root)

        # Initialize the rest
        super().__init__(root=root, *args, **kwargs)

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        instances = make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )
        index = self.index_classes(instances, class_to_idx=class_to_idx)

        weighted_instances = []
        for class_ in self.classes:
            # Calculate number of samples (based on current population)
            class_weight = self.class_weights[class_]
            target_num_samples = int(len(instances) * class_weight)

            # Sample with replacement (allows oversampling)
            class_idx = self.class_to_idx[class_]
            sample_idxs = index[class_idx]
            weighted_sample_idxs = self.rng.choices(sample_idxs, k=target_num_samples)

            weighted_instances += [instances[idx] for idx in weighted_sample_idxs]

        return weighted_instances

    @staticmethod
    def index_classes(
        instances: List[Tuple[str, int]], class_to_idx: Dict[str, int]
    ) -> List[List[int]]:
        index = [[] for _ in range(len(class_to_idx))]
        for idx, (_, class_) in enumerate(instances):
            index[class_].append(idx)

        return index
