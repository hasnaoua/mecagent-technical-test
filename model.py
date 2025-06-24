from transformers import VisionEncoderDecoderModel, AutoTokenizer
from typing import Tuple

def encoder_decoder_model(
    encode_path: str, 
    decoder_path: str
) -> Tuple[VisionEncoderDecoderModel, AutoTokenizer]:
    """
    Loads a VisionEncoderDecoderModel with specified encoder and decoder pretrained models,
    configures decoder start token, pad token, and vocab size, and returns the model and tokenizer.

    Args:
        encode_path (str): Pretrained model name or path for the encoder.
        decoder_path (str): Pretrained model name or path for the decoder.

    Returns:
        Tuple[VisionEncoderDecoderModel, AutoTokenizer]: The configured model and tokenizer.
    """
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encode_path,
        decoder_pretrained_model_name_or_path=decoder_path
    )

    # Load tokenizer from encoder path (you might want to load from decoder_path if it makes more sense)
    tokenizer = AutoTokenizer.from_pretrained(encode_path)

    # Configure special tokens for the decoder
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Set vocab size explicitly from decoder config
    model.config.vocab_size = model.config.decoder.vocab_size

    return model, tokenizer
