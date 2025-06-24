from prepare_data import load_data, process_dataset
from model import encoder_decoder_model
from train import train_model
from transformers import BlipProcessor

def main():
    encoder_model_path = "Salesforce/blip-image-captioning-base"
    decoder_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_path = "./checkpoints/baseline"

    print("Loading dataset...")
    dataset = load_data()

    print("Loading processor and models...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model, tokenizer = encoder_decoder_model(encoder_model_path, decoder_model_path)

    print("Preprocessing dataset...")
    train_dataset, eval_dataset = process_dataset(dataset, processor, tokenizer)

    print("Starting training...")
    train_model(train_dataset, eval_dataset, tokenizer, model, output_path)

    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
