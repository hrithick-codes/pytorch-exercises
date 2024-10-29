import re
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from loguru import logger

from app.config import Config
from utils.helpers import get_device

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")


@dataclass
class Hyperparams:
    num_context_words: int = 32
    vocab_size: int = 13257 + 1
    hidden_dim: int = 256
    batch_size: int = 2048
    num_epochs: int = 10000

    oov_token_id: int = 13258
    oov_token: str = "<UNK>"
    device: str = get_device()
    splitting_pattern: str = r"\w+|[^\w\s]|\s+"

    train_size: float = 0.8
    val_size: float = 0.1


class Word2VecCBOW(nn.Module):
    def __init__(
        self, num_context_words: int, vocab_size: int, hidden_dim: int
    ):  # noqa: E501
        super(Word2VecCBOW, self).__init__()
        self._vocab_size = vocab_size
        self._num_context_words = num_context_words

        weights = torch.zeros(size=(self._vocab_size, hidden_dim))
        initialized_weights = torch.nn.init.xavier_uniform_(weights)
        self._context_layers = nn.Parameter(data=initialized_weights)

        self._out_layer = nn.Linear(
            in_features=num_context_words, out_features=self._vocab_size
        )

    def forward(self, x):
        size = tuple(x.size())
        assert size[1:] == (
            self._num_context_words,
            self._vocab_size,
        ), f"x should be of size (context_word, vocab_size) but got size {x.size()}"  # noqa: E501
        hidden_representation = torch.einsum(
            "bcv,vh->bch", x, self._context_layers
        )  # noqa: E501
        latent_vector = torch.sum(hidden_representation, dim=-1)
        logits = self._out_layer(latent_vector)
        return logits


hyperparams = Hyperparams()
model = Word2VecCBOW(
    hyperparams.num_context_words,
    hyperparams.vocab_size,
    hyperparams.hidden_dim,
)
model = model.to(hyperparams.device)
# Does not support MPS as of now! :(
# model = torch.compile(model)
config = Config()


logger.info("Reading the text corpus...")
with open(config.GOT_BOOK_LANGUAGE_MODELLING, "r") as file:
    content = file.read()
content = content.lower()
logger.info("Done")


def split_into_tokens(text):
    return re.findall(hyperparams.splitting_pattern, text)


def build_vocab(
    corpus: List, vocab_size: int, oov_token: str, oov_token_id: int
):  # noqa: E501
    all_tokens = split_into_tokens(corpus)
    logger.info(f"Total tokens in dataset: {len(set(all_tokens)) + 1}")
    freq_tokens = Counter(all_tokens).most_common(vocab_size)
    token_to_id = {
        token: iid for iid, (token, _) in enumerate(freq_tokens, start=0)  # noqa
    }
    id_to_token = {id: token for token, id in token_to_id.items()}

    token_to_id[oov_token] = oov_token_id
    id_to_token[oov_token_id] = oov_token
    return token_to_id, id_to_token


def encode(text):
    tokens = split_into_tokens(text)
    indices = list(
        map(lambda x: token_to_id.get(x, hyperparams.oov_token_id), tokens)
    )  # noqa: E501
    return indices


def decode(ids):
    return "".join([id_to_token.get(id, hyperparams.oov_token) for id in ids])


logger.info("Building vocabulary...")
token_to_id, id_to_token = build_vocab(
    content,
    hyperparams.vocab_size,
    hyperparams.oov_token,
    hyperparams.oov_token_id,
)
logger.info("Converting into ids...")
indices = encode(content)
logger.info("Done.")


logger.info("Split into train and test...")
num_tokens = len(indices)
train_tokens = int(num_tokens * hyperparams.train_size)
val_tokens = int(num_tokens * hyperparams.val_size)
test_tokens = int(
    num_tokens * (1 - hyperparams.train_size - hyperparams.val_size)
)  # noqa: E501


train_indices = indices[0:train_tokens]
val_indices = indices[train_tokens : train_tokens + val_tokens]  # noqa
test_indices = indices[-test_tokens]
logger.info("Done.")


def get_batch(indices):
    start_indices = torch.randint(
        low=0,
        high=len(indices) - hyperparams.num_context_words,
        size=(hyperparams.batch_size,),
    )

    context_words = []
    target_words = []

    for start_index in start_indices:
        context_tokens = indices[
            start_index : start_index  # noqa: E203
            + hyperparams.num_context_words
            + 1  # noqa: E203 E501
        ]
        # Choose and pop the focus word
        focus_token_id = start_index + (hyperparams.num_context_words // 2)
        focus_token = indices[focus_token_id]
        # Remove focus token
        context_tokens.remove(focus_token)

        context_words.append(context_tokens)
        target_words.append(focus_token)

    context_words = torch.tensor(context_words)
    one_hot_context_words = torch.nn.functional.one_hot(
        context_words, hyperparams.vocab_size
    ).float()

    target_words = torch.tensor(target_words, dtype=torch.long)

    return one_hot_context_words, target_words


context_tensor, target_tensor = get_batch(indices)
print("Context Tensor Shape:", context_tensor.shape)
print("Target Tensor Shape:", target_tensor.shape)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())


for epoch in range(hyperparams.num_epochs):
    model.train()
    # Get training batch
    context, target = get_batch(train_indices)

    context = context.to(hyperparams.device)
    target = target.to(hyperparams.device)

    optimizer.zero_grad()
    outputs = model(context)

    loss = loss_function(outputs, target)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        # Get validation batch
        context, target = get_batch(val_indices)
        context = context.to(hyperparams.device)
        target = target.to(hyperparams.device)
        outputs = model(context)
        val_loss = loss_function(outputs, target)

    print(
        f"Epoch {epoch + 1}/{10}, Training Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"  # noqa: E501
    )
