import torch
from torch import nn
import random
from collections import Counter
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from loguru import logger

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
    TRAIN_SIZE: float = 0.8
    VAL_SIZE: float = 0.1
    VOCAB_SIZE: int = 100000
    OOV_TOKEN: str = "<UNK>"
    OOV_TOKEN_ID: int = 0
    BATCH_SIZE: int = 32
    MAX_LENGHT: int = 512


hyperparams = Hyperparams()


def build_vocab(tokens, vocab_size, oov_token, oov_token_id):
    """Build a vocabulary and assigned id to token based on frequency"""
    freq_tokens = Counter(tokens).most_common(vocab_size)
    token_to_id = {
        token: iid for iid, (token, count) in enumerate(freq_tokens, start=1)
    }
    id_to_token = {id: token for token, id in token_to_id.items()}

    token_to_id[oov_token] = oov_token_id
    id_to_token[oov_token_id] = oov_token
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
    hyperparams.OOV_TOKEN_ID,  # noqa
)
logger.info("Done")


def detokenize(ids, id_to_token=id_to_token):
    tokens = []
    for token_id in ids:
        token = id_to_token.get(token_id, hyperparams.OOV_TOKEN_ID)
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
logger.info('Loading dataloaders...')
train_dataloader = DataLoader(
    train_dataset, batch_size=hyperparams.BATCH_SIZE, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=hyperparams.BATCH_SIZE, collate_fn=collate_fn
)
test_dataloader = DataLoader(
    test_dataset, batch_size=hyperparams.BATCH_SIZE, collate_fn=collate_fn
)
logger.info('Done')


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super(RNNClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer with `n_layers` and hidden size `hidden_dim`
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Embedding lookup
        embedded = self.embedding(x)
        
        # RNN returns hidden states and last hidden state
        rnn_out, hidden = self.rnn(embedded)
        
        # Get the last hidden state from the final RNN layer
        final_hidden = hidden[-1]  # Take the last layer's hidden state
        
        # Pass through the fully connected layer
        output = self.fc(final_hidden)
        
        # Sigmoid for binary classification
        output = self.sigmoid(output)
        
        return output