# Transformer Block

## Why do we have multiple Blocks? What do they calculate, what are the differences?

Each Block does the same *type* of work — attend, then compute — but with completely independent weights. Stacking them lets each block refine the representation produced by the one before it.

Think of it like a pipeline of editors. The first editor reads the raw text and produces a better draft. The second reads that draft (not the original) and improves it further. By the end, the representation has been refined 4 times.

### What one Block calculates

A single Block does two things in sequence:

**Attention (communication)**
Each token looks at all previous tokens and asks: *given what I know about my neighbours, how should I update my own representation?* The result is a weighted blend of other tokens' values. Information flows between positions.

**FFN (computation)**
Each token position is then transformed independently through a small MLP. No mixing with neighbours — just a pointwise nonlinear transformation on the blended representation.

The residual connections (`x = x + ...`) mean each block is not replacing `x` but adding a correction to it.

### Differences between the 4 Blocks

Architecturally: nothing. They are identical in structure and hyperparameters. Every block has:
- The same 4 attention heads with `head_size = n_embd // n_head = 16`
- The same FFN hidden size `4 * n_embd = 256`
- Its own independent set of weights (Q, K, V, projection, FFN — all separate)

Functionally, specialization is emergent, not designed. Early blocks tend toward local/syntactic patterns; later blocks toward longer-range structure.

---

## Are Blocks the same as layers?

Yes. **Blocks are layers** in the transformer sense. In transformer literature "layer" and "block" are used interchangeably. When someone says "GPT-2 has 12 layers," each of those layers is one Block: one attention + one FFN + their LayerNorms and residuals.

`n_layer = 4` → 4 transformer layers.

---

## Every Block has its own weights, hasn't it?

Yes. Each of the 4 blocks owns independent weights:

| Component | Weights |
|---|---|
| `ln1` | scale + bias (`n_embd` each) |
| `sa` — 4 × Head | Q, K, V matrices (`n_embd × head_size` each) |
| `sa` — projection | `n_embd × n_embd` |
| `ln2` | scale + bias |
| `ffwd` — linear 1 | `n_embd × 4*n_embd` |
| `ffwd` — linear 2 | `4*n_embd × n_embd` |

None of these are shared. You can verify it from the trace log — each object has a different memory address. Different address = independent weights.

---

## What kind of weights are ln1 and ln2? They are type nn.LayerNorm.

`nn.LayerNorm` has two learnable weight tensors:

- **`weight`** (γ, gamma) — scale vector, shape `(n_embd,)`, initialized to all 1s
- **`bias`** (β, beta) — shift vector, shape `(n_embd,)`, initialized to all 0s

What LayerNorm does in the forward pass:
1. **Normalize** — subtract mean, divide by std across the embedding dimension (no learned parameters here)
2. **Rescale** — multiply elementwise by γ
3. **Shift** — add elementwise by β

γ and β let the model undo the normalization if needed. Without them, LayerNorm would force every layer to always work with zero-mean unit-variance inputs, which is too restrictive.

In this model `n_embd = 64`, so each `ln1` and `ln2` has 64 + 64 = **128 learnable parameters** — small compared to attention and FFN. Their job is stabilization, not representation.

---

## What does FeedForward (ffwd linear 1 and 2) do?

The FFN is a two-layer MLP applied independently to each token position.

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # expand:   64 → 256
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),   # contract: 256 → 64
    nn.Dropout(dropout),
)
```

**Linear 1** expands from `n_embd` to `4 * n_embd`. The wider middle layer gives the model more working space to compute intermediate features.

**ReLU** introduces nonlinearity — zeroes out negatives. Without it, the two linear layers would collapse into one.

**Linear 2** projects back to `n_embd` to fit back into the residual stream.

After attention, each token holds a blended representation — a linear combination of its neighbours. The FFN does **nonlinear processing** on that result. The mental model: attention decides *what information to gather*, FFN decides *what to do with it*.

Key point: unlike attention, FFN processes each token independently. The same weights are applied to every position. No communication between positions here — that all happened in attention.

---

## Lines 168 and 175 are interesting. Why x = x + sa_out and x = x + ffwd_out?

These are the **residual connections** (also called skip connections).

```python
x = x + sa_out    # attention adds a correction, does not replace x
x = x + ffwd_out  # FFN adds a further correction
```

Without the residual it would be `x = sa_out` — the output **replaces** `x` entirely. With the residual, each sub-layer only contributes a **correction** (delta) on top of what was already there.

Across a full 4-block pass, `x` accumulates refinements:

```
x                      ← raw embedding (token + position)
x = x + Δ_attn_1      ← block 1 attention correction
x = x + Δ_ffn_1       ← block 1 FFN correction
x = x + Δ_attn_2      ← block 2 attention correction
x = x + Δ_ffn_2       ← ...
...
```

### Why this matters

**Gradient flow during training**
In a deep network without residuals, gradients shrink as they are multiplied back through each layer (vanishing gradient). With residuals there is a direct path from the loss all the way back to early layers — the gradient can flow through the `+` unchanged. This is what made training very deep networks practical.

**Each block is more independent**
Without residuals, if block 4 learns something wrong it poisons the gradient signal for blocks 1–3. With residuals, each block can be trained to add useful information without needing to relay everything that came before.

**Identity as a safe baseline**
If a block's weights are near zero, `x = x + ~0 ≈ x`. The block becomes a near-identity function. This means a deeper network is never *worse* than a shallower one at initialization — extra layers can always be learned away.

### The key intuition

Think of `x` as a **running document** and each block as an editor who writes margin notes. The margin notes (`sa_out`, `ffwd_out`) are added to the document. The original text is never erased. By the final block, `x` is the original document plus all the accumulated annotations from every layer.
