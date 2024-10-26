import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import List

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.config import Config
from utils.helpers import get_device

random.seed(1337)
torch.manual_seed(1337)

config = Config()

logger.info("Reading datasets...")
with open(config.TEXT_CLASSIFICATION_Y, "r") as file:
    labels = file.read()

with open(config.TEXT_CLASSIFICATION_X, "r") as file:
    reviews = file.read()

logger.info("Splitting into samples...")
sequences = reviews.lower().split("\n")
labels = labels.lower().split("\n")

logger.info("Shuffling the samples...")
together = list(zip(sequences, labels))
random.shuffle(together)
sequences, labels = zip(*together)
logger.info("All done!")


@dataclass
class Hyperparams:
    TRAIN_SIZE: float = 0.6
    VAL_SIZE: float = 0.2
    VOCAB_SIZE: int = 500000
    OOV_TOKEN: str = "<UNK>"
    OOV_TOKEN_ID: int = 1
    PAD_TOKEN: str = "<PAD>"
    PAD_TOKEN_ID: int = 0
    BATCH_SIZE: int = 1024
    MAX_LENGHT: int = 20

    # Neural network arch configs
    EMBEDDING_DIM: int = 512
    RNN_NEURONS: int = 128
    RNN_LAYERS: int = 4
    RNN_DROPOUT: float = 0.3
    LINEAR_NEURONS: int = 256
    LEARNING_RATE: float = 0.001
    OVERFIT_SINGLE_BATCH: bool = False
    EPOCHS: int = 10

    # Hardware
    DEVICE = get_device()
    WARM_UP_ON_GPU: bool = True
    MODEL_SAVE_PATH: str = "models/sarcasm_rnn.pth"


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
        for iid, (token, count) in enumerate(
            freq_tokens, start=oov_token_id + 1
        )  # noqa: E501
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

val_sequences = sequences[train_size : train_size + val_size]  # noqa
val_labels = labels[train_size : train_size + val_size]  # noqa

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
        return [1 if label == "true" else 0 for label in labels]

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
    )  # noqa: E501
    for i, token_ids in enumerate(inputs):
        length = len(token_ids)
        padded_inputs[i, :length] = torch.tensor(
            token_ids[: hyperparams.MAX_LENGHT]
        )  # noqa: E501

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
        self._hidden_dim = rnn_hidden_size
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
            dropout=hyperparams.RNN_DROPOUT,
            bias=False,
        )
        self._linear_linear = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_neurons_in_linear,
            bias=False,  # noqa: E501
        )
        self._relu = nn.ReLU()
        self._output_layer = nn.Linear(
            in_features=num_neurons_in_linear, out_features=1, bias=False
        )

    def forward(self, x):
        x = self._embedding_layer(x)
        hidden_states, out = self._rnn(x)
        last_hidden_state = hidden_states[:, -1, :]
        linear_out = self._linear_linear(last_hidden_state)
        linear_out = self._relu(linear_out)
        logits = self._output_layer(linear_out)
        return logits.squeeze()


logger.info("Initialize the model...")
model = RNNClassifier(
    vocab_size=hyperparams.VOCAB_SIZE,
    embedding_dim=hyperparams.EMBEDDING_DIM,
    rnn_hidden_size=hyperparams.RNN_NEURONS,
    num_neurons_in_linear=hyperparams.LINEAR_NEURONS,
)
logger.info(model)
logger.info("Moving the model to {hyperparams.DEVICE}")
model = model.to(hyperparams.DEVICE)
logger.info("Compiling using torch.compile()...")
model = torch.compile(model)
logger.info("Done!")

# Put the model on training mode
model.train()

# Use Binary cross entropy loss
loss_function = nn.BCEWithLogitsLoss()

# Adam for optimizing the loss function
optim = torch.optim.Adam(model.parameters(), lr=hyperparams.LEARNING_RATE)


# Overfit a batch
batch = next(iter(train_dataloader))
input_var, output_var = batch
# Move both the tensors into device
input_var = input_var.to(hyperparams.DEVICE)
output_var = output_var.to(hyperparams.DEVICE)

# Perform a warm up
if hyperparams.WARM_UP_ON_GPU and hyperparams.DEVICE in ("cuda", "mps"):
    logger.info(f"Warming up the model on {hyperparams.DEVICE}...")
    for i in range(15):
        _ = model(input_var)
    logger.info("Done")


if hyperparams.OVERFIT_SINGLE_BATCH:
    logger.info("Overfitting a single batch...")
    start = time.time()
    bce_loss = float("inf")
    iter_count = 1
    print(input_var)
    while bce_loss > 0.05:
        logits = model(input_var)
        bce_loss = loss_function(logits, output_var.float())
        print("Iteration: ", iter_count, "Loss: ", round(bce_loss.item(), 2))
        bce_loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()
        optim.zero_grad()
        iter_count += 1
    print(
        f"Batch overfitted. Time Taken: {time.time() - start}. Took {iter_count} iterations."  # noqa
    )
    logger.info("Done")
    import sys

    sys.exit()

# Enhanced Training and validation loop with batch-level details
logger.info("Starting training loop...")

best_val_accuracy = -float("inf")
for epoch in range(hyperparams.EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0.0
    correct_train_preds = 0
    total_train_preds = 0

    print(
        f"Epoch [{epoch+1}/{hyperparams.EPOCHS}] - Starting training phase..."
    )  # noqa: E501

    for batch_idx, batch in enumerate(train_dataloader):
        inputs, targets = batch
        inputs = inputs.to(hyperparams.DEVICE)
        targets = targets.to(hyperparams.DEVICE)

        # Forward pass
        logits = model(inputs)
        loss = loss_function(logits, targets.float())

        # Backward pass and optimization
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()

        # Batch loss and accuracy
        total_train_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_train_preds += (preds == targets).sum().item()
        total_train_preds += targets.size(0)

        # Log batch-level details
        if batch_idx % 10 == 0:
            avg_batch_loss = total_train_loss / (batch_idx + 1)
            batch_accuracy = correct_train_preds / total_train_preds
            print(
                f"Batch [{batch_idx+1}/{len(train_dataloader)}], Batch Loss: {avg_batch_loss:.4f}, Batch Accuracy: {batch_accuracy:.4f}"  # noqa: E501
            )

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = correct_train_preds / total_train_preds
    print(
        f"Epoch [{epoch+1}/{hyperparams.EPOCHS}] completed. Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"  # noqa: E501
    )

    # Validation phase
    model.eval()
    total_val_loss = 0.0
    correct_val_preds = 0
    total_val_preds = 0

    print(
        "Epoch [{epoch+1}/{hyperparams.EPOCHS}] - Starting validation phase..."
    )  # noqa: E501

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs, targets = batch
            inputs = inputs.to(hyperparams.DEVICE)
            targets = targets.to(hyperparams.DEVICE)

            # Forward pass
            logits = model(inputs)
            loss = loss_function(logits, targets.float())

            # Batch loss and accuracy
            total_val_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_val_preds += (preds == targets).sum().item()
            total_val_preds += targets.size(0)

            # Log batch-level validation details
            if batch_idx % 10 == 0:
                avg_val_batch_loss = total_val_loss / (batch_idx + 1)
                val_batch_accuracy = correct_val_preds / total_val_preds
                print(
                    f"Epoch [{epoch+1}/{hyperparams.EPOCHS}], Val Batch [{batch_idx+1}/{len(val_dataloader)}], "  # noqa: E501
                    f"Val Batch Loss: {avg_val_batch_loss:.4f}, Val Batch Accuracy: {val_batch_accuracy:.4f}"  # noqa: E501
                )

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = correct_val_preds / total_val_preds

    if val_accuracy > best_val_accuracy:
        logger.info(
            f"Validation accuracy increased from {best_val_accuracy} to {val_accuracy}"  # noqa: E501
        )
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), hyperparams.MODEL_SAVE_PATH)
        logger.info("Model saved.")

    print(
        f"Epoch [{epoch+1}/{hyperparams.EPOCHS}] completed. Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"  # noqa: E501
    )

logger.info("Validating on unseen data...")
with torch.no_grad():
    correct_val_preds = 0.0
    total_preds = 0.0
    for batch_idx, batch in enumerate(test_dataloader):
        inputs, targets = batch
        inputs = inputs.to(hyperparams.DEVICE)
        targets = targets.to(hyperparams.DEVICE)

        # Forward pass
        logits = model(inputs)

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_val_preds += (preds == targets).sum().item()
        total_preds += len(targets)
    test_acc = correct_val_preds / total_preds

logger.info(f"Test accuracy: {test_acc}")
logger.info("Training complete!")

# Ensure model is in evaluation mode
model.eval()
while True:
    input_string = input("Enter the text: ")
    tokens = input_string.strip().split(" ")
    input_ids = [
        token_to_id.get(token, hyperparams.OOV_TOKEN_ID) for token in tokens
    ]  # noqa: E501
    input_tensor = torch.tensor(input_ids).unsqueeze(dim=0)
    input_tensor = input_tensor.to(hyperparams.DEVICE)
    print("Tokens: ", tokens)
    print("Input ids: ", input_tensor)
    logits = model(input_tensor)
    sigmoid_output = torch.sigmoid(logits).item()
    print(f"Sigmoid Output: {sigmoid_output:.4f}")
