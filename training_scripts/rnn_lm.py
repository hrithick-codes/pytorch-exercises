import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
from loguru import logger
from torch import nn

from app.common import DATASETS, PATHS, choose_device

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")

logger.info("Reading the text corpus...")
with open(DATASETS.TINY_SHAKESPERE_DATASET, "r") as file:
    content = file.read()
logger.info("Done")


@dataclass
class Hyperparams:
    VOCAB_SIZE: int = 13339
    SPLITTING_PATTERN: str = r"\w+|[^\w\s]|\s+"
    OOV_TOKEN: str = "<UNK>"
    OOV_TOKEN_ID: int = 1
    BLOCK_SIZE: int = 64
    DO_LOWERCASE: str = False

    EMBEDDING_DIM: int = 256
    HIDDEN_SIZE: int = 256
    NUM_LAYERS: int = 16
    LABEL_SMOOTHING: float = 0.1
    LEARNING_RATE: float = 0.001

    STEPS: int = 4500
    BATCH_SIZE: int = 256
    DEVICE: str = choose_device()


hyperparams = Hyperparams()
if hyperparams.DO_LOWERCASE:
    logger.info("Converting into lower case...")
    content = content.lower()
    logger.info("Done.")


def split_into_tokens(text):
    return re.findall(hyperparams.SPLITTING_PATTERN, text)


def build_vocab(corpus: List, vocab_size: int, oov_token: str, oov_token_id: int):
    """Build a vocabulary and assigned id to token based on frequency"""
    # Tokenization to get tokens from text
    all_tokens = split_into_tokens(corpus)
    logger.info(f"Total tokens in dataset: {len(set(all_tokens))}")
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
    content,
    hyperparams.VOCAB_SIZE,
    hyperparams.OOV_TOKEN,
    hyperparams.OOV_TOKEN_ID,
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


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
    ):
        super(LanguageModel, self).__init__()
        self._embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self._recurrent_model = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=hyperparams.NUM_LAYERS,
            batch_first=True,
            bias=True,
        )

        self._fc = nn.Linear(hidden_size, vocab_size, bias=False)
        self._fc.weight = self._embedding_layer.weight

    def forward(self, x):
        x = self._embedding_layer(x)
        hidden_states, _ = self._recurrent_model(x)
        logits = self._fc(hidden_states)
        return logits


logger.info("Initializing the model...")
model = LanguageModel(
    hyperparams.VOCAB_SIZE, hyperparams.EMBEDDING_DIM, hyperparams.HIDDEN_SIZE
)
model = model.to(hyperparams.DEVICE)
model = torch.compile(model)
logger.info("Done")

print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.LEARNING_RATE)
criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams.LABEL_SMOOTHING)

logger.info(f"Choosing batch size: {hyperparams.BATCH_SIZE}...")
logger.info("Starting training...")

x_, y_ = get_batch()
print("Input")
print(x_)
print("Target")
print(y_)


for step in range(1, hyperparams.STEPS + 1):
    model.train()
    start = time.time()
    x, y = get_batch()
    x = x.to(hyperparams.DEVICE)
    y = y.to(hyperparams.DEVICE)

    logits = model(x)

    # Reshape target and output for loss computation
    logits = logits.view(-1, hyperparams.VOCAB_SIZE)
    target = y.view(-1)

    loss = criterion(logits, target)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    optimizer.zero_grad()

    tok_per_sec = int(torch.numel(x) // (time.time() - start))

    print(
        f"Step: {step}|{hyperparams.STEPS}, Loss: {loss.item()}, Perplexity: {torch.exp(loss)}, Tok/sec: {tok_per_sec}"
    )

    model.eval()
    with torch.no_grad():
        random_token = random.choice(list(token_to_id.keys()))
        generated_tokens = [token_to_id.get(random_token, hyperparams.OOV_TOKEN_ID)]
        print("Prefix: ", random_token)
        for pos in range(hyperparams.BLOCK_SIZE):
            generated_tensor = (
                torch.tensor(generated_tokens).unsqueeze(0).to(hyperparams.DEVICE)
            )
            hidden_states = model(generated_tensor)
            logits = hidden_states[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            predicted_token_id = torch.multinomial(probabilities, num_samples=1).item()
            predicted_token = id_to_token.get(predicted_token_id, hyperparams.OOV_TOKEN)
            generated_tokens.append(predicted_token_id)

        print("Sampled text:", decode(generated_tokens), "\n")


logger.info("Saving the model and tokenizer dict...")
torch.save(model.state_dict(), PATHS.RECURRENT_LM_SAVE_PATH)

with open(PATHS.RECURRENT_LM_TOKENIZER_SAVE_PATH, "w") as file:
    json.dump(token_to_id, file, indent=4)

logger.info("Saved! Exiting...")
