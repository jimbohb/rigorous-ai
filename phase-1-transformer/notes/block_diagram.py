"""
Visualize the internal structure of a Transformer Block.

Draws the journey of the input tensor through one Block, with tensor
dimensions annotated at every step. Uses only matplotlib so the drawing
logic stays readable.

Run:  python block_diagram.py   ->  writes block_diagram.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- hyperparameters (mirrors instrumented.py) ---------------------------
B = 16   # batch_size
T = 32   # block_size  (sequence length)
C = 64   # n_embd      (embedding dim)
n_head = 4
head_size = C // n_head   # 16


def dim(*shape):
    """Format a tensor shape like (B, T, C)."""
    return "(" + ", ".join(str(s) for s in shape) + ")"


# Colours grouped by role so related steps read as a family.
COL_IO = "#cfe8ff"      # input / output
COL_NORM = "#fff2cc"    # layernorm
COL_ATTN = "#d5e8d4"    # attention
COL_FFN = "#f8cecc"     # feed-forward
COL_ADD = "#e1d5e7"     # residual add

fig, ax = plt.subplots(figsize=(11, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 24)
ax.axis("off")

# Main flow runs down the centre column; residual stream sits on the right.
XC = 3.2          # centre x of the main pipeline boxes
X_RES = 8.2       # x of the residual stream line
BOX_W = 4.4
BOX_H = 1.0


def box(y, label, sublabel, color, w=BOX_W, x=XC):
    """Draw a labelled box centred at (x, y). Returns (top_y, bottom_y)."""
    rect = FancyBboxPatch(
        (x - w / 2, y - BOX_H / 2), w, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4, edgecolor="#333333", facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.13, label, ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(x, y - 0.22, sublabel, ha="center", va="center",
            fontsize=9, color="#444444", family="monospace")
    return y + BOX_H / 2, y - BOX_H / 2


def arrow(y_from, y_to, x=XC, label=None, color="#333333"):
    """Vertical arrow from y_from down to y_to along column x."""
    a = FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color=color,
    )
    ax.add_patch(a)
    if label:
        ax.text(x + 0.3, (y_from + y_to) / 2, label, ha="left", va="center",
                fontsize=8, color="#666666", family="monospace")


# --- the pipeline, top to bottom -----------------------------------------
# Each entry: (label, sublabel, color). We lay them out at fixed y steps.
ax.text(XC, 23.3, "Transformer Block", ha="center", fontsize=15,
        fontweight="bold")
ax.text(XC, 22.7,
        f"B={B} (batch)   T={T} (seq len)   C={C} (embd)   "
        f"{n_head} heads x {head_size}",
        ha="center", fontsize=9, color="#555555", family="monospace")

y = 21.5

# input
top, bot = box(y, "input  x", dim(B, T, C), COL_IO)
y_input_bot = bot

# ln1
y -= 2.0
arrow(y_input_bot, y + BOX_H / 2)
top, bot = box(y, "LayerNorm  ln1", dim(B, T, C), COL_NORM)
y_ln1_bot = bot

# --- attention sub-block (drawn as a grouped region) ---
y -= 2.3
attn_top = y + BOX_H / 2 + 0.5
arrow(y_ln1_bot, y + BOX_H / 2 + 0.7)

# Multi-head attention internals box (Q/K/V -> scores -> aggregate)
top, bot = box(y, "MultiHeadAttention", "4 heads in parallel", COL_ATTN,
               w=BOX_W)
# detail lines inside/under the attention header
y -= 1.4
ax.text(XC, y + 0.35,
        f"per head:  Q,K,V = Linear(C -> {head_size})   ->  "
        f"q,k,v {dim(B, T, head_size)}",
        ha="center", fontsize=8, family="monospace", color="#2b5a2b")
ax.text(XC, y - 0.0,
        f"scores  wei = q @ kᵀ / √{head_size}   ->  {dim(B, T, T)}",
        ha="center", fontsize=8, family="monospace", color="#2b5a2b")
ax.text(XC, y - 0.35,
        f"out = softmax(wei) @ v   ->  {dim(B, T, head_size)}",
        ha="center", fontsize=8, family="monospace", color="#2b5a2b")
y -= 1.1
ax.text(XC, y + 0.1,
        f"concat 4 heads -> {dim(B, T, C)}   then proj Linear(C->C)",
        ha="center", fontsize=8, family="monospace", color="#2b5a2b")
attn_bot = y - 0.35

# draw a dashed grouping rectangle around the attention region
group = FancyBboxPatch(
    (XC - BOX_W / 2 - 0.25, attn_bot - 0.15),
    BOX_W + 0.5, attn_top - attn_bot + 0.3,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    linewidth=1.0, edgecolor="#82b366", facecolor="none", linestyle="--",
)
ax.add_patch(group)

# residual add 1
y -= 1.4
arrow(attn_bot - 0.15, y + BOX_H / 2)
top, bot = box(y, "+  residual add", "x = x + sa_out  " + dim(B, T, C),
               COL_ADD)
y_add1 = y

# ln2
y -= 2.0
arrow(bot, y + BOX_H / 2)
top, bot = box(y, "LayerNorm  ln2", dim(B, T, C), COL_NORM)
y_ln2_bot = bot

# --- FFN sub-block ---
y -= 2.3
ffn_top = y + BOX_H / 2 + 0.5
arrow(y_ln2_bot, y + BOX_H / 2 + 0.7)
top, bot = box(y, "FeedForward", "position-wise MLP", COL_FFN)
y -= 1.4
ax.text(XC, y + 0.35,
        f"Linear(C -> 4C)   ->  {dim(B, T, 4 * C)}",
        ha="center", fontsize=8, family="monospace", color="#a02020")
ax.text(XC, y - 0.0, "ReLU  (nonlinearity)",
        ha="center", fontsize=8, family="monospace", color="#a02020")
ax.text(XC, y - 0.35,
        f"Linear(4C -> C)   ->  {dim(B, T, C)}",
        ha="center", fontsize=8, family="monospace", color="#a02020")
ffn_bot = y - 0.35
group2 = FancyBboxPatch(
    (XC - BOX_W / 2 - 0.25, ffn_bot - 0.15),
    BOX_W + 0.5, ffn_top - ffn_bot + 0.3,
    boxstyle="round,pad=0.02,rounding_size=0.1",
    linewidth=1.0, edgecolor="#b85450", facecolor="none", linestyle="--",
)
ax.add_patch(group2)

# residual add 2
y -= 1.4
arrow(ffn_bot - 0.15, y + BOX_H / 2)
top, bot = box(y, "+  residual add", "x = x + ffwd_out  " + dim(B, T, C),
               COL_ADD)
y_add2 = y

# output
y -= 2.0
arrow(bot, y + BOX_H / 2)
top, bot = box(y, "output  x", dim(B, T, C), COL_IO)

# --- residual stream (skip connections) ----------------------------------
# Branch off below input, bypass attention, feed into add1.
def residual_path(y_branch, y_join, label):
    # out to the right, down, back in
    ax.plot([XC + BOX_W / 2, X_RES], [y_branch, y_branch],
            color="#9673a6", linewidth=1.6, linestyle="-")
    ax.plot([X_RES, X_RES], [y_branch, y_join],
            color="#9673a6", linewidth=1.6, linestyle="-")
    a = FancyArrowPatch(
        (X_RES, y_join), (XC + BOX_W / 2, y_join),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color="#9673a6",
    )
    ax.add_patch(a)
    ax.text(X_RES + 0.15, (y_branch + y_join) / 2, label,
            ha="left", va="center", fontsize=8, color="#9673a6", rotation=90)


residual_path(y_input_bot - 0.1, y_add1, "skip (identity)")
residual_path(y_add1 - BOX_H / 2 - 0.1, y_add2, "skip (identity)")

plt.tight_layout()
out = "block_diagram.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
