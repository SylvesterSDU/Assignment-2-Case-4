"""Train a tiny next-token transformer on fineweb_corpus.txt (CPU, about 60 seconds).

Run from the folder that holds fineweb_corpus.txt and fineweb_tokens.txt:
    python3 train.py

What it does
  1. Loads the corpus and your token IDs, and checks them against GPT-4o's o200k_base tokenizer.
     If they do not match, it regenerates them from the corpus.
  2. Remaps the token IDs that occur in the corpus to a small internal vocabulary (0..V-1),
     so the model only has to choose between tokens it has actually seen.
  3. Trains a tiny, randomly initialised causal transformer to predict the next token.
  4. Prints live lines: step | loss | elapsed time | tokens/second.
  5. Before training, every 20 updates and at the end, generates text from the same short
     sentence beginning with the same random seed, streaming tokens as they appear.
  6. Saves: outputs/training_log.csv, outputs/loss_plot.png, outputs/samples.txt, outputs/model.pt
"""
import csv
import math
import os
import sys
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------- settings
CORPUS_FILE = "fineweb_corpus.txt"
TOKENS_FILE = "fineweb_tokens.txt"
OUT_DIR = "outputs"
TRAIN_SECONDS = 60          # training time budget
CONTEXT = 64                # tokens the model sees at once
BATCH = 16
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
LR = 3e-3
SAMPLE_EVERY = 20           # generate a sample every N updates
SAMPLE_TOKENS = 30          # tokens per sample
SAMPLE_SEED = 1234          # same seed for every sample, so samples are comparable
START_TEXT = "If you have"  # short sentence beginning (occurs in the corpus)

torch.manual_seed(0)
torch.set_num_threads(max(1, os.cpu_count() or 1))
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------- 1. data and tokenizer
enc = tiktoken.get_encoding("o200k_base")
corpus = open(CORPUS_FILE, encoding="utf-8").read()
expected = enc.encode(corpus)

with open(TOKENS_FILE, encoding="utf-8") as f:
    given = [int(x) for x in f.read().replace(",", " ").split()]

if given == expected:
    print(f"Token check: OK. {len(given)} token IDs in {TOKENS_FILE} match the corpus.")
    ids = given
else:
    print(f"Token check: MISMATCH ({len(given)} given vs {len(expected)} from the corpus). "
          f"Using token IDs regenerated from {CORPUS_FILE}.")
    ids = expected

# ----------------------------------------------------------------- 2. compact vocabulary
vocab = sorted(set(ids))                      # original o200k IDs that occur in the corpus
to_small = {tok: i for i, tok in enumerate(vocab)}
to_big = {i: tok for tok, i in to_small.items()}
V = len(vocab)
data = torch.tensor([to_small[t] for t in ids], dtype=torch.long)
print(f"Corpus: {len(corpus)} characters, {len(ids)} tokens, {V} different tokens "
      f"(the full GPT-4o vocabulary has {enc.n_vocab}).")

start_ids = enc.encode(START_TEXT)
missing = [t for t in start_ids if t not in to_small]
if missing:
    print(f"'{START_TEXT}' contains tokens not in the training vocabulary; "
          f"using the first 3 corpus tokens as the beginning instead.")
    start_ids = ids[:3]
    START_TEXT = enc.decode(start_ids)
start_small = [to_small[t] for t in start_ids]


def get_batch():
    ix = torch.randint(0, len(data) - CONTEXT - 1, (BATCH,))
    x = torch.stack([data[i:i + CONTEXT] for i in ix])
    y = torch.stack([data[i + 1:i + CONTEXT + 1] for i in ix])
    return x, y


# ----------------------------------------------------------------- 3. tiny transformer
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # causal: no peeking ahead
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D_MODEL)
        self.pos = nn.Embedding(CONTEXT, D_MODEL)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYERS)])
        self.ln = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, V)

    def forward(self, idx):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        return self.head(self.ln(self.blocks(x)))


model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_LAYERS} layers, {D_MODEL} dimensions, {n_params:,} parameters, randomly initialised.")
print(f"Loss of pure guessing among {V} tokens: {math.log(V):.2f}\n")

samples_file = open(os.path.join(OUT_DIR, "samples.txt"), "w", encoding="utf-8")


@torch.no_grad()
def generate(label):
    """Generate SAMPLE_TOKENS tokens from START_TEXT with a fixed seed, streaming them."""
    model.eval()
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    idx = torch.tensor([start_small], dtype=torch.long)
    out_big = list(start_ids)
    shown = enc.decode(out_big)
    print(f"--- sample ({label}) ---")
    sys.stdout.write(shown)
    sys.stdout.flush()
    for _ in range(SAMPLE_TOKENS):
        logits = model(idx[:, -CONTEXT:])[:, -1, :]
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g)
        idx = torch.cat([idx, nxt], dim=1)
        out_big.append(to_big[nxt.item()])
        text = enc.decode(out_big)
        sys.stdout.write(text[len(shown):])   # stream only the new part
        sys.stdout.flush()
        shown = text
    print("\n")
    samples_file.write(f"[{label}]\n{shown}\n\n")
    samples_file.flush()
    model.train()


# ----------------------------------------------------------------- 4. training loop
log = []
generate("before training, step 0")
print("step | loss | elapsed time | tokens/second")
t0 = time.time()
step = 0
while time.time() - t0 < TRAIN_SECONDS:
    x, y = get_batch()
    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += 1
    elapsed = time.time() - t0
    tps = step * BATCH * CONTEXT / elapsed
    log.append((step, loss.item(), elapsed, tps))
    print(f"{step:5d} | {loss.item():.3f} | {elapsed:6.1f}s | {tps:,.0f}", flush=True)
    if step % SAMPLE_EVERY == 0:
        generate(f"step {step}")

generate(f"end of training, step {step}")
samples_file.close()

# ----------------------------------------------------------------- 5. save results
with open(os.path.join(OUT_DIR, "training_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "loss", "elapsed_seconds", "tokens_per_second"])
    w.writerows(log)

torch.save({"model_state": model.state_dict(), "vocab": vocab,
            "config": dict(V=V, CONTEXT=CONTEXT, D_MODEL=D_MODEL, N_HEADS=N_HEADS, N_LAYERS=N_LAYERS)},
           os.path.join(OUT_DIR, "model.pt"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    plt.plot([r[0] for r in log], [r[1] for r in log])
    plt.axhline(math.log(V), linestyle="--", linewidth=1, label=f"pure guessing ({math.log(V):.2f})")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Training loss of the tiny model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_plot.png"), dpi=150)
except ImportError:
    print("matplotlib not installed: loss plot skipped.")

print(f"Done: {step} steps in {time.time() - t0:.0f} seconds. Files saved in '{OUT_DIR}/':")
print("  training_log.csv, loss_plot.png, samples.txt, model.pt")
