# Next Token Prediction

## How it works

### Training (`forward`)

1. The index of the last token selects a row from the embedding table — this row contains the weights (raw logits). Logits can be any real number, positive or negative.
2. Softmax converts the logits into a probability vector. All values are between 0 and 1 and sum to 1.
3. `targets[i]` tells us which position in the probability vector should have the highest value. If that position has a small probability, the loss is high.
4. Because B×T predictions are computed in parallel, `F.cross_entropy` averages all losses into a single scalar.
5. The optimizer performs gradient descent — it nudges the embedding weights so the correct token's probability increases, reducing the loss.

### Inference (`generate`)

The model samples from the probability vector. Instead of checking against a target, it picks the next token randomly weighted by the probabilities. Higher probability → more likely to be chosen, but any token can be picked.

---

## Source code

### Training data generation

```python
import numpy as np

def generate_training_data_special_pattern(size: int) -> str:
    data_int = []
    i = 0
    while i < size - 1:
        item = ord("a")              # every character is 'a' in this simplified version
        data_int.append(item)
        i += 1
        if item == ord('a'):
            i += 1
            random2 = np.random.rand()
            if random2 <= 0.5:
                data_int.append(ord("b"))   # 'a' → 'b' with 50% probability
            elif random2 <= 0.7:
                data_int.append(ord("c"))   # 'a' → 'c' with 20% probability
            elif random2 <= 0.8:
                data_int.append(ord("d"))   # 'a' → 'd' with 10% probability
            else:
                data_int.append(int(np.random.randint(ord('e'), ord('z') + 1)))  # other with 20%
    if i != size:
        data_int.append(int(np.random.randint(ord('b'), ord('z') + 1)))
    return "".join([chr(x) for x in data_int])
```

The training data has a deliberate pattern: `'a'` is always followed by a non-random character. After training, the model should learn these probabilities from the data.

### Vocabulary

```python
class Vocabulary:
    def __init__(self, data):
        self.vocabulary = sorted(list(set(list(data))))   # unique characters, sorted
        self.stoi = { c:i for i,c in enumerate(self.vocabulary) }  # char → index
        self.itos = { i:c for i,c in enumerate(self.vocabulary) }  # index → char

    def encode_data(self, input: str) -> list:
        return [self.stoi[x] for x in input]   # string → list of integers

    def decode_data(self, input: list) -> str:
        return "".join([self.itos[x] for x in input])  # list of integers → string

    def size(self) -> int:
        return len(self.vocabulary)
```

### Model

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class LLM_Forward_Loss_Generator(nn.Module):

    def __init__(self, vocabulary: Vocabulary, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # embedding table: shape (vocab_size, vocab_size)
        # each row = raw logits for the next token given this token
        self.token_embedding_table = nn.Embedding(vocabulary.size(), EMBEDDING_DIM)

    def forward(self, idx, targets=None):
        # idx: (B, T) — token indices
        # embedding lookup: each index → a row of logits
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            # inference mode — no loss needed
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)   # flatten to (N, C) for cross_entropy
            targets = targets.view(B * T)    # flatten to (N,)
            # cross_entropy:
            #   1. applies softmax to logits → probabilities
            #   2. picks the probability at targets[i] for each prediction
            #   3. computes -log(that probability)
            #   4. averages across all B*T predictions
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```

### Training loop (in progress)

```python
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)

for steps in range(max_iters):
    xb, yb = get_batch('train')       # input and target sequences
    logits, loss = network(xb, yb)    # forward pass, compute loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                   # backpropagation — compute gradients
    optimizer.step()                  # update embedding weights
```

### Generation (sampling)

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)               # forward pass, no targets
        logits = logits[:, -1, :]             # take only the last token's logits (B, C)
        probs = F.softmax(logits, dim=-1)     # convert to probabilities (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # sample one token (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)             # append and repeat
    return idx
```

`torch.multinomial` picks a token randomly, weighted by probabilities. A token with probability 0.5 gets chosen roughly half the time. This produces varied output — always picking the highest probability (argmax) would cause the model to loop.
