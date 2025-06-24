from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, VisionEncoderDecoderModel
from typing import Optional

def train_model(
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizer,
    model: VisionEncoderDecoderModel,
    training_args: Optional[TrainingArguments] = None,
    output_path: str = "./checkpoints/genCAD-baseline",
    tokenizer_output_path: Optional[str] = None
):
    """
    Train the encoder-decoder model with given datasets, tokenizer, and training arguments.

    Args:
        train_dataset: Dataset for training.
        eval_dataset: Dataset for evaluation.
        tokenizer (PreTrainedTokenizer): Tokenizer used for the model.
        model (VisionEncoderDecoderModel): The encoder-decoder model to train.
        training_args (TrainingArguments, optional): Training arguments. If None, default args are used.
        output_path (str): Path to save the trained model.
        tokenizer_output_path (str, optional): Path to save the tokenizer. Defaults to output_path if None.

    Returns:
        None
    """
    if training_args is None:
        training_args = TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            save_strategy="steps",
            num_train_epochs=1,
            save_steps=500,
            eval_steps=500,
            logging_steps=100,
            learning_rate=1e-5,
            fp16=True,
            report_to="none"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Optional but recommended
    )

    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(tokenizer_output_path or output_path)
