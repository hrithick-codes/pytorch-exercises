import torch


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available, using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print(
            "MPS (Metal Performance Shaders) is available, \
            using Apple Silicon GPU."
        )
        return torch.device("mps")
    else:
        print("Neither CUDA nor MPS is available, using CPU.")
        return torch.device("cpu")
