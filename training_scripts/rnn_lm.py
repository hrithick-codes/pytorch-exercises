import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
from loguru import logger
from torch import nn

from app.common import DATASETS, choose_device

torch.manual_seed(1337)

logger.info("Reading the text corpus...")
with open(DATASETS.GOT_BOOK_LANGUAGE_MODELLING, "r") as file:
    content = file.read()
logger.info("Done")


@dataclass
class Hyperparams:
    VOCAB_SIZE: int = 24000
    SPLITTING_PATTERN: str = r"\w+|[^\w\s]|\s+"
    OOV_TOKEN: str = "<UNK>"
    OOV_TOKEN_ID: int = 1
    BLOCK_SIZE: int = 64

    EMBEDDING_DIM: int = 256
    RNN_HIDDEN_SIZE: int = 256
    RNN_LAYERS: int = 8
    LABEL_SMOOTHING: float = 0.1

    STEPS: int = 10000
    BATCH_SIZE: int = 256
    DEVICE: str = choose_device()


hyperparams = Hyperparams()


def split_into_tokens(text):
    return re.findall(hyperparams.SPLITTING_PATTERN, text)


def build_vocab(corpus: List, vocab_size: int, oov_token: str, oov_token_id: int):
    """Build a vocabulary and assigned id to token based on frequency"""
    # Tokenization to get tokens from text
    all_tokens = split_into_tokens(corpus)
    # Choosing the topK tokens, topK is the vocab_size
    freq_tokens = Counter(all_tokens).most_common(vocab_size)
    # Each tokens will have an unique id
    token_to_id = {
        token: iid for iid, (token, _) in enumerate(freq_tokens, start=oov_token_id + 1)
    }
    # Reverse mapping
    id_to_token = {id: token for token, id in token_to_id.items()}

    # Add OOV token
    token_to_id[oov_token] = oov_token_id
    id_to_token[oov_token_id] = oov_token
    return token_to_id, id_to_token


logger.info("Building vocabulary...")
token_to_id, id_to_token = build_vocab(
    content, hyperparams.VOCAB_SIZE, hyperparams.OOV_TOKEN, hyperparams.OOV_TOKEN_ID
)
logger.info("Done.")


def encode(text):
    tokens = split_into_tokens(text)
    return torch.tensor(
        list(map(lambda x: token_to_id.get(x, hyperparams.OOV_TOKEN_ID), tokens)),
        dtype=torch.long,
    )


def decode(ids):
    return "".join([id_to_token.get(id, hyperparams.OOV_TOKEN) for id in ids])


input_ids = encode(content)


def get_batch():
    indices = torch.randint(
        low=0,
        high=len(input_ids) - hyperparams.BLOCK_SIZE,
        size=(hyperparams.BATCH_SIZE,),
    )
    x = torch.stack([input_ids[i : i + hyperparams.BLOCK_SIZE] for i in indices])
    y = torch.stack(
        [input_ids[i + 1 : i + hyperparams.BLOCK_SIZE + 1] for i in indices]
    )
    return x, y


class RNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn_hidden_size: int,
    ):
        super(RNNLanguageModel, self).__init__()
        self._embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        # self._rnn = nn.RNN(
        #     input_size=embedding_dim,
        #     hidden_size=rnn_hidden_size,
        #     num_layers=hyperparams.RNN_LAYERS,
        #     batch_first=True,
        #     nonlinearity="relu",
        #     bias=False,
        # )
        self._rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=hyperparams.RNN_LAYERS,
            batch_first=True,
            bias=False,
        )

        self._fc = nn.Linear(rnn_hidden_size, vocab_size, bias=False)
        # share weights between the fc and embedding layer
        self._fc.weight = self._embedding_layer.weight

    def forward(self, x):
        x = self._embedding_layer(x)
        hidden_states, _ = self._rnn(x)
        logits = self._fc(hidden_states)
        return logits


logger.info("Initializing the model...")
model = RNNLanguageModel(
    hyperparams.VOCAB_SIZE, hyperparams.EMBEDDING_DIM, hyperparams.RNN_HIDDEN_SIZE
)
model = model.to(hyperparams.DEVICE)
model = torch.compile(model)
logger.info("Done")

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams.LABEL_SMOOTHING)


for step in range(1, hyperparams.STEPS + 1):
    model.train()
    x, y = get_batch()
    x = x.to(hyperparams.DEVICE)
    y = x.to(hyperparams.DEVICE)

    logits = model(x)

    # Reshape target and output for loss computation
    logits = logits.view(-1, hyperparams.VOCAB_SIZE)
    target = y.view(-1)

    loss = criterion(logits, target)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(
        f"Step: {step}|{hyperparams.STEPS}, Loss: {loss.item()}, Perplexity: {torch.exp(loss)}"
    )

    model.eval()
    with torch.no_grad():
        random_token = random.choice(
            [
                "Winter",
                "Dragon",
                "Throne",
                "Targaryen",
                "Stark",
                "Lannister",
                "Night",
                "Walkers",
                "Jon",
                "Daenerys",
                "King",
                "Queen",
                "Lord",
            ]
        )
        generated_tokens = [token_to_id.get(random_token, hyperparams.OOV_TOKEN_ID)]
        print("Prefix: ", random_token)
        for _ in range(hyperparams.BLOCK_SIZE):
            generated_tensor = (
                torch.tensor(generated_tokens).unsqueeze(0).to(hyperparams.DEVICE)
            )
            inference_output = model(generated_tensor)
            predicted_token_id = torch.argmax(inference_output[:, -1, :], dim=-1).item()
            predicted_token = id_to_token.get(predicted_token_id, hyperparams.OOV_TOKEN)
            generated_tokens.append(predicted_token_id)

        print("Sampled text:", decode(generated_tokens))
