import random
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
from loguru import logger
from torch import nn

from app.common import DATASETS, batch_data, choose_device

logger.info("Reading the text corpus...")
with open(DATASETS.GOT_BOOK_LANGUAGE_MODELLING, "r") as file:
    content = file.readlines()

logger.info("Done! Starting to preprocess...")
stripped_content = list(map(lambda x: x.strip(), content))
removed_spaces = list(filter(lambda x: x, stripped_content))
corpus = " ".join(removed_spaces)
tokens = corpus.lower().split(" ")
logger.info(f"Done. Number of tokens: {len(set(tokens))}")


@dataclass
class Hyperparams:
    TRAIN_SIZE: float = 0.6
    VAL_SIZE: float = 0.2
    VOCAB_SIZE: int = 24000
    OOV_TOKEN: str = "<UNK>"
    OOV_TOKEN_ID: int = 1
    BATCH_SIZE: int = 2048
    MAX_LENGHT: int = 10

    EMBEDDING_DIM: int = 512
    RNN_HIDDEN_SIZE: int = 512
    RNN_LAYERS: int = 1
    LABEL_SMOOTHING: float = 0.1

    EPOCHS: int = 10000
    DEVICE: str = choose_device()


hyperparams = Hyperparams()


def build_vocab(tokens: List, vocab_size: int, oov_token: str, oov_token_id: int):
    """Build a vocabulary and assigned id to token based on frequency"""
    freq_tokens = Counter(tokens).most_common(vocab_size)
    token_to_id = {
        token: iid
        for iid, (token, count) in enumerate(freq_tokens, start=oov_token_id + 1)
    }
    id_to_token = {id: token for token, id in token_to_id.items()}

    # add oov token
    token_to_id[oov_token] = oov_token_id
    id_to_token[oov_token_id] = oov_token
    return freq_tokens, token_to_id, id_to_token


logger.info("Building vocabulary...")
vocab, token_to_id, id_to_token = build_vocab(
    tokens, hyperparams.VOCAB_SIZE, hyperparams.OOV_TOKEN, hyperparams.OOV_TOKEN_ID
)
logger.info("Done.")


def generate_dataset(tokens, chunk_size, token_to_id):
    source = []
    target = []
    for source_start_point in range(0, len(tokens) - chunk_size - 1):
        target_start_point = source_start_point + 1
        source_tokens = tokens[source_start_point : source_start_point + chunk_size]
        target_tokens = tokens[target_start_point : target_start_point + chunk_size]

        # convert into tokens
        source_input_ids = [
            token_to_id.get(token, hyperparams.OOV_TOKEN_ID) for token in source_tokens
        ]
        target_token_ids = [
            token_to_id.get(token, hyperparams.OOV_TOKEN_ID) for token in target_tokens
        ]
        source.append(source_input_ids)
        target.append(target_token_ids)
    return [source, target]


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
        self._rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=hyperparams.RNN_LAYERS,
            batch_first=True,
            nonlinearity="relu",
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

logger.info("Generating the datasets...")
source_input_ids, target_input_ids = generate_dataset(
    tokens, hyperparams.MAX_LENGHT, token_to_id
)
logger.info("Done")
logger.info("Starting training...")

for epoch in range(1, hyperparams.EPOCHS + 1):
    step = 0
    for input_id, target in batch_data(
        source_input_ids, target_input_ids, hyperparams.BATCH_SIZE
    ):
        input_id = input_id.to(hyperparams.DEVICE)
        target = target.to(hyperparams.DEVICE)

        # Forward pass through the model
        logits = model(input_id)

        # Reshape target and output for loss computation
        logits = logits.view(
            -1, hyperparams.VOCAB_SIZE
        )  # (batch_size * seq_len, vocab_size)
        target = target.view(-1)  # (batch_size * seq_len)

        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        print(
            f"Epoch[{epoch}|{hyperparams.EPOCHS}], Step: {step}, Loss: {loss.item()}, Perplexity: {torch.exp(loss)}"
        )

        # Inference
        model.eval()
        with torch.no_grad():
            random_token = random.choice(tokens)
            generated_tokens = [token_to_id.get(random_token, hyperparams.OOV_TOKEN_ID)]
            print("Prefix: ", random_token)
            for _ in range(hyperparams.MAX_LENGHT):
                generated_tensor = (
                    torch.tensor(generated_tokens).unsqueeze(0).to(hyperparams.DEVICE)
                )
                inference_output = model(generated_tensor)
                predicted_token_id = torch.argmax(
                    inference_output[:, -1, :], dim=-1
                ).item()
                predicted_token = id_to_token[predicted_token_id]
                generated_tokens.append(predicted_token_id)

            generated_text = [
                id_to_token.get(token, hyperparams.OOV_TOKEN)
                for token in generated_tokens
            ]
            print("Sampled text:", " ".join(generated_text), "\n")
