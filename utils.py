import matplotlib.pyplot as plt
from typing import List
import random




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
