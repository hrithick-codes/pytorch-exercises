from dataclasses import dataclass

import torch


def choose_device():
    if torch.cuda.is_available():
        print("CUDA is available, using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        print("Neither CUDA nor MPS is available, using CPU.")
        return torch.device("cpu")


@dataclass
class DATASETS:
    TEXT_CLASSIFICATION_X: str = "data/sarcasm_detection/texts.txt"
    TEXT_CLASSIFICATION_Y: str = "data/sarcasm_detection/labels.txt"
