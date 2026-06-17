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
    """Every pair is 'a' followed by b/c/d/e-z with fixed probabilities."""
    data_int = []
    i = 0
    while i < size - 1:
        data_int.append(ord('a'))
        i += 1
        r = np.random.rand()
        if r <= 0.50:
            data_int.append(ord('b'))   # 'a' → 'b' with 50% probability
        elif r <= 0.70:
            data_int.append(ord('c'))   # 'a' → 'c' with 20% probability
        elif r <= 0.80:
            data_int.append(ord('d'))   # 'a' → 'd' with 10% probability
        else:
            data_int.append(int(np.random.randint(ord('e'), ord('z') + 1)))  # other with 20%
        i += 1
    if i < size:
        data_int.append(int(np.random.randint(ord('b'), ord('z') + 1)))
    return "".join(chr(x) for x in data_int)
```

The training data has a deliberate pattern: `'a'` is always followed by a non-random character. After training, the model should learn these probabilities from the data.

### Vocabulary

```python
class Vocabulary:
    def __init__(self, data: str):
        self.vocabulary = sorted(set(data))                        # unique characters, sorted
        self.stoi = {c: i for i, c in enumerate(self.vocabulary)}  # char → index
        self.itos = {i: c for i, c in enumerate(self.vocabulary)}  # index → char

    def encode_data(self, text: str) -> list:
        return [self.stoi[x] for x in text]    # string → list of integers

    def decode_data(self, tokens: list) -> str:
        return "".join(self.itos[x] for x in tokens)  # list of integers → string

    def size(self) -> int:
        return len(self.vocabulary)
```

### Model

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab: Vocabulary):
        super().__init__()
        vocab_size = vocab.size()
        # Rows = current token index; values = logits over next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) — token indices
        # embedding lookup: each index → a row of logits
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        # cross_entropy:
        #   1. applies softmax to logits → probabilities
        #   2. picks the probability at targets[i] for each prediction
        #   3. computes -log(that probability)
        #   4. averages across all B*T predictions
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss
```

### Train / validation split

```python
n = int(0.9 * len(data))   # 90% train, 10% validation
train_data = data[:n]
val_data = data[n:]
```

### Data loading

```python
BATCH_SIZE = 32
BLOCK_SIZE = 8

def get_batch(split: str):
    src = train_data if split == 'train' else val_data
    ix = torch.randint(len(src) - BLOCK_SIZE, (BATCH_SIZE,))       # random start positions
    x = torch.stack([src[i:i + BLOCK_SIZE]     for i in ix])       # input: (B, T)
    y = torch.stack([src[i + 1:i + BLOCK_SIZE + 1] for i in ix])   # target: shifted by 1
    return x, y
```

Targets are the inputs shifted one position forward — each token predicts the next one.

### Training loop

```python
optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)

for step in range(10_000):
    xb, yb = get_batch('train')               # input and target sequences
    logits, loss = network(xb, yb)            # forward pass, compute loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                           # backpropagation — compute gradients
    optimizer.step()                          # update embedding weights

# final loss: ~0.965
```

Loss dropped from ~4.24 (random) to ~0.965 after 10 000 steps.

### Generation (sampling)

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens: int):
    # idx: (B, T) — running context; we extend it one token at a time
    for _ in range(max_new_tokens):
        logits, _ = self(idx)                              # forward pass, no targets
        probs = F.softmax(logits[:, -1, :], dim=-1)        # last token's probabilities (B, vocab_size)
        idx_next = torch.multinomial(probs, num_samples=1) # sample one token (B, 1)
        idx = torch.cat([idx, idx_next], dim=1)            # append and repeat
    return idx
```

`torch.multinomial` picks a token randomly, weighted by probabilities. A token with probability 0.5 gets chosen roughly half the time. This produces varied output — always picking the highest probability (argmax) would cause the model to loop.

### Verifying the learned pattern

After training, 1000 tokens were generated and the bigram frequencies counted:

```
a   → 1000  (every token is 'a' in the simplified training data)
ab  →  523  (≈ 50% ✓)
ac  →  203  (≈ 20% ✓)
ad  →   87  (≈ 10% ✓)
ae  →    9  (≈ rare ✓)
```

The model learned the pattern from data — no pattern was hard-coded into the architecture.

### Inspecting the learned weights

After training, the embedding row for `'a'` (index 0) directly encodes the next-token distribution:

```python
w0 = network.token_embedding_table.weight.data[0]  # raw logits for 'a'
sm = F.softmax(w0, dim=-1)                          # → probabilities
```

Plotting `sm` shows a bar chart with high values at indices for `'b'`, `'c'`, `'d'` and low values elsewhere — the probability table the model converged to matches the training distribution.
