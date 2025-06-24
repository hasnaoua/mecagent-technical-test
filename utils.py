from datasets import load_dataset
import matplotlib.pyplot as plt
from typing import List
import random


def load_data(dataset_path: str = "CADCODER/GenCAD-Code") -> dict:
    """
    Loads the specified dataset using the Hugging Face datasets library.

    Args:
        dataset_path (str): Path or identifier of the dataset.

    Returns:
        dict: A dictionary containing train and test splits.
    """
    ds = load_dataset(
        dataset_path,
        num_proc=16,
        split=["train", "test"],
    )
    return ds


def plot_samples(dataset: List[dict], samples_count: int = 5, randomize: bool = False) -> None:
    """
    Plots sample images and prints corresponding prompt and CadQuery code.

    Args:
        dataset (List[dict]): The dataset to sample from.
        samples_count (int): Number of samples to display.
        randomize (bool): Whether to select random samples.
    """
    indices = (
        random.sample(range(len(dataset)), samples_count)
        if randomize else range(samples_count)
    )

    for i in indices:
        sample = dataset[i]
        deepcad_id = sample.get('deepcad_id', f"Sample {i}")
        image = sample['image']
        prompt = sample.get('prompt', "No prompt available.")
        code = sample.get('cadquery', "No CadQuery code available.")

        # Show image
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Sample ID: {deepcad_id}")
        plt.show()

        # Show corresponding CadQuery code
        print(f"\n=== Prompt and CadQuery Code for Sample ID: {deepcad_id} ===")
        print("Prompt:")
        print(prompt)
        print("\nCadQuery Code:")
        print(code)
        print("=" * 60)
