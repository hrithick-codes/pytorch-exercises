from dataclasses import dataclass


@dataclass
class Config:
    """
    A configuration class to hold file paths for datasets and model storage,
    used across various NLP tasks such as text classification and language
    modeling.
    """

    # Text classification datasets
    TEXT_CLASSIFICATION_X: str = "data/sarcasm_detection/texts.txt"
    TEXT_CLASSIFICATION_Y: str = "data/sarcasm_detection/labels.txt"
    # Language modelling
    GOT_BOOK_LANGUAGE_MODELLING: str = "data/got/GOT.txt"
    TINY_SHAKESPERE_DATASET: str = "data/shakespere/tiny_shakespere.txt"

    # Model save paths
    RECURRENT_LM_SAVE_PATH: str = "models/rnn-lm.pth"
    RECURRENT_LM_TOKENIZER_SAVE_PATH: str = "models/tokenizer.json"
