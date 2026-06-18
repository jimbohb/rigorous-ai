import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import threading

# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
LoadFromFile = True
LogEnable = False
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def mylog(self, event: str, indent="", name=None):
    global LogEnable
    if LogEnable:
        label = name if name else type(self).__name__
        print(
            f"{indent}{threading.get_native_id()}:{hex(id(self))}:{label}:{event}"
        )


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mylog(self, "start", "\t\t\t")
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        mylog(self, "end", "\t\t\t")
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mylog(self, "start", "\t\t")
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        mylog(self, "end", "\t\t")
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        mylog(self, "start", "\t\t")
        retval = self.net(x)
        mylog(self, "end", "\t\t")
        return retval


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, indent="\t"):
        mylog(self, "start", indent=indent)
        mylog(self, "start", indent=indent + "\t", name="ln1")
        normed = self.ln1(x)
        mylog(self, "end", indent=indent + "\t", name="ln1")
        sa_out = self.sa(normed)
        mylog(self, "start", indent=indent + "\t", name="residual_add_1")
        x = x + sa_out
        mylog(self, "end", indent=indent + "\t", name="residual_add_1")
        mylog(self, "start", indent=indent + "\t", name="ln2")
        normed = self.ln2(x)
        mylog(self, "end", indent=indent + "\t", name="ln2")
        ffwd_out = self.ffwd(normed)
        mylog(self, "start", indent=indent + "\t", name="residual_add_2")
        x = x + ffwd_out
        mylog(self, "end", indent=indent + "\t", name="residual_add_2")
        mylog(self, "end", indent=indent)
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        mylog(self, "start")
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        mylog(self, "start", "\t", name="token_embedding")
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        mylog(self, "end", "\t", name="token_embedding")
        mylog(self, "start", "\t", name="position_embedding")
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        mylog(self, "end", "\t", name="position_embedding")
        mylog(self, "start", "\t", name="embedding_add")
        x = tok_emb + pos_emb  # (B,T,C)
        mylog(self, "end", "\t", name="embedding_add")
        x = self.blocks(x)  # (B,T,C)
        mylog(self, "start", "\t", name="ln_f")
        x = self.ln_f(x)  # (B,T,C)
        mylog(self, "end", "\t", name="ln_f")
        mylog(self, "start", "\t", name="lm_head")
        logits = self.lm_head(x)  # (B,T,vocab_size)
        mylog(self, "end", "\t", name="lm_head")

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        mylog(self, "end")
        return logits, loss

    def generate(self, idx, max_new_tokens):
        global LogEnable
        # idx is (B, T) array of indices in the current context
        for index in range(max_new_tokens):
            if index >= 0:
                LogEnable = True
            else:
                LogEnable = False
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
# print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

if LoadFromFile and os.path.exists("gpt_model.pt"):
    m.load_state_dict(torch.load("gpt_model.pt", weights_only=True))
    # print("Loaded weights from gpt_model.pt — skipping training.")
else:
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
torch.manual_seed(42)
# print(decode(m.generate(context, max_new_tokens=1)[0].tolist()))
m.generate(context, max_new_tokens=1)[0].tolist()
