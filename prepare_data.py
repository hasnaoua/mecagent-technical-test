import torch
from datasets import DatasetDict
from datasets import load_dataset
from transformers import PreTrainedTokenizer, PreTrainedProcessor




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



def preprocess_sample(
    sample: dict,
    processor: PreTrainedProcessor,
    tokenizer: PreTrainedTokenizer,
) -> dict:
    """
    Preprocess a single sample by encoding the image and tokenizing the code text.

    Args:
        sample (dict): A dictionary with 'image' and 'code' keys.
        processor (PreTrainedProcessor): Image processor (e.g., CLIPProcessor).
        tokenizer (PreTrainedTokenizer): Tokenizer for code (e.g., GPT2Tokenizer).

    Returns:
        dict: Dictionary containing encoded image and tokenized labels.
    """
    image = sample["image"]
    text = sample["code"]

    # Encode image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

    # Tokenize text
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    labels = tokenized.input_ids.squeeze(0)

    # Replace padding token id with -100 for loss masking
    labels = torch.where(labels == tokenizer.pad_token_id, torch.tensor(-100), labels)

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


def process_dataset(
    dataset: DatasetDict,
    processor: PreTrainedProcessor,
    tokenizer: PreTrainedTokenizer
):
    """
    Process the train and test datasets by applying the preprocessing function.

    Args:
        dataset (DatasetDict): Hugging Face DatasetDict with 'train' and 'test' splits.
        processor (PreTrainedProcessor): Image processor.
        tokenizer (PreTrainedTokenizer): Tokenizer for text.

    Returns:
        tuple: Processed train and evaluation datasets.
    """
    train_dataset = dataset["train"].map(
        lambda sample: preprocess_sample(sample, processor, tokenizer),
        remove_columns=dataset["train"].column_names
    )

    eval_dataset = dataset["test"].map(
        lambda sample: preprocess_sample(sample, processor, tokenizer),
        remove_columns=dataset["test"].column_names
    )

    return train_dataset, eval_dataset
