import random
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

random.seed(1337)


logger.info("Reading datasets...")
with open("scripts/data/labels.txt", "r") as file:
    labels = file.read()

with open("scripts/data/reviews.txt", "r") as file:
    reviews = file.read()

logger.info("Splitting into samples...")
sequences = reviews.lower().split("\n")
labels = labels.lower().split("\n")

logger.info("Shuffling the samples...")
together = list(zip(sequences, labels))
random.shuffle(together)
sequences, labels = zip(*together)
logger.info("All done!")


@dataclass  # noqa: F821
class Hyperparams:
    TRAIN_SIZE: float = 0.6
    VAL_SIZE: float = 0.2
    VOCAB_SIZE: int = 50000
    OOV_TOKEN: str = "<UNK>"
    OOV_TOKEN_ID: int = 1
    PAD_TOKEN: str = "<PAD>"
    PAD_TOKEN_ID: int = 0
    BATCH_SIZE: int = 32
    MAX_LENGHT: int = 128

    # Neural network arch configs
    EMBEDDING_DIM: int = 64
    RNN_NEURONS: int = 256
    RNN_LAYERS: int = 2
    LINEAR_NEURONS: int = 128
    LEARNING_RATE: float = 0.005


hyperparams = Hyperparams()


def build_vocab(
    tokens: List,
    vocab_size: int,
    oov_token: str,
    oov_token_id: int,
    pad_token: str,
    pad_token_id: int,
):
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
    # add padding token
    token_to_id[pad_token] = pad_token_id
    id_to_token[pad_token_id] = pad_token
    return freq_tokens, token_to_id, id_to_token


logger.info("Splitting into train-test-val split...")
num_samples = len(sequences)
train_size = int(num_samples * hyperparams.TRAIN_SIZE)
val_size = int(num_samples * hyperparams.VAL_SIZE)
test_size = num_samples - (train_size + val_size)

# Load dataset split into train-test-split
train_sequences = sequences[0:train_size]
train_labels = labels[0:train_size]

val_sequences = sequences[train_size : train_size + val_size]
val_labels = labels[train_size : train_size + val_size]

test_sequences = sequences[-test_size:]
test_labels = labels[-test_size:]
logger.info("Done.")

# Flatten all the tokens
word_tokens = []
for seq in train_sequences:
    for word in seq.split(" "):
        word_tokens.append(word)

# Build vocab only using training dataset
logger.info("Building vocab using training dataset...")
vocab, token_to_id, id_to_token = build_vocab(
    word_tokens,
    hyperparams.VOCAB_SIZE,
    hyperparams.OOV_TOKEN,
    hyperparams.OOV_TOKEN_ID,
    hyperparams.PAD_TOKEN,
    hyperparams.PAD_TOKEN_ID,
)
logger.info("Done")


def detokenize(ids, id_to_token=id_to_token):
    tokens = []
    for token_id in ids:
        token = id_to_token.get(token_id, hyperparams.OOV_TOKEN)
        tokens.append(token)
    return " ".join(tokens)


class SentimentDataset(Dataset):
    """Dataset class for classification"""

    def __init__(self, sequences, labels, token_to_id, oov_token_id):
        self.input_ids = self.tokenize(sequences, token_to_id, oov_token_id)
        self.labels = self.convert_labels_to_numeric(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index]

    def convert_labels_to_numeric(self, labels):
        return [1 if label == "positive" else 0 for label in labels]

    def tokenize(self, sequences, token_to_id, oob_token_id):
        input_ids = []
        for seq in sequences:
            tokens = seq.strip().split(" ")
            ids = []
            for token in tokens:
                current_id = token_to_id.get(token, oob_token_id)
                ids.append(current_id)
            input_ids.append(ids)
        return input_ids


train_dataset = SentimentDataset(
    train_sequences, train_labels, token_to_id, hyperparams.OOV_TOKEN_ID
)
val_dataset = SentimentDataset(
    val_sequences, val_labels, token_to_id, hyperparams.OOV_TOKEN_ID
)
test_dataset = SentimentDataset(
    test_sequences, test_labels, token_to_id, hyperparams.OOV_TOKEN_ID
)

logger.info(f"Training samples: {len(train_dataset)}")
logger.info(f"Validation samples: {len(val_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")


def collate_fn(batch):
    """Perform padding and truncation in the batch"""
    inputs, targets = zip(*batch)
    num_samples = len(inputs)
    # padding the inputs
    padded_inputs = torch.zeros(
        num_samples, hyperparams.MAX_LENGHT, dtype=torch.long
    )  # noqa
    for i, token_ids in enumerate(inputs):
        length = len(token_ids)
        padded_inputs[i, :length] = torch.tensor(
            token_ids[: hyperparams.MAX_LENGHT]
        )  # noqa

    return padded_inputs, torch.tensor(targets, dtype=torch.float32)


# Define the dataloaders
logger.info("Loading dataloaders...")
train_dataloader = DataLoader(
    train_dataset, batch_size=hyperparams.BATCH_SIZE, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=hyperparams.BATCH_SIZE, collate_fn=collate_fn
)
test_dataloader = DataLoader(
    test_dataset, batch_size=hyperparams.BATCH_SIZE, collate_fn=collate_fn
)
logger.info("Done")


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn_hidden_size: int,
        num_neurons_in_linear: int,
    ):
        super(RNNClassifier, self).__init__()
        self._embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=hyperparams.PAD_TOKEN_ID,
        )
        self._rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_hidden_size,
            num_layers=hyperparams.RNN_LAYERS,
            batch_first=True,
            nonlinearity="relu",
        )
        self._linear_linear = nn.Linear(
            in_features=rnn_hidden_size, out_features=num_neurons_in_linear
        )
        self._relu = nn.ReLU()
        self._output_layer = nn.Linear(
            in_features=num_neurons_in_linear, out_features=1
        )

    def forward(self, x):
        x = self._embedding_layer(x)
        hidden_states, out = self._rnn(x)
        last_hidden_state = hidden_states[:, -1, :]
        linear_out = self._linear_linear(last_hidden_state)
        linear_out = self._relu(linear_out)
        logits = self._output_layer(linear_out)
        return logits.squeeze()


model = RNNClassifier(
    vocab_size=hyperparams.VOCAB_SIZE,
    embedding_dim=hyperparams.EMBEDDING_DIM,
    rnn_hidden_size=hyperparams.RNN_NEURONS,
    num_neurons_in_linear=hyperparams.LINEAR_NEURONS,
)

# Put the model on training mode
model.train()

# Use Binary cross entropy loss
loss_function = nn.BCEWithLogitsLoss()

# Adam for optimizing the loss function
optim = torch.optim.Adam(model.parameters(), lr=hyperparams.LEARNING_RATE)


# Overfit a batch
batch = next(iter(train_dataloader))
input_var, output_var = batch

while True:
    logits = model(input_var)
    bce_loss = loss_function(logits, output_var.float())
    if bce_loss.item() < 0.05:
        print("Loss: ", round(bce_loss.item(), 2))
        print("Batch overfitted!")
        break
    print("Loss: ", round(bce_loss.item(), 2))
    bce_loss.backward()
    # Clip gradients for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optim.step()
    optim.zero_grad()
    

