"""Pick a short sentence that appears exactly in fineweb_corpus.txt, use its first few words as
the beginning, and let the trained model complete it 100 times.

Run from the 4. Build-A-LLM folder (with the .venv active):
    python3 corpus_test.py

The script checks that every token of the beginning exists in the model's saved vocabulary,
and chooses another sentence if needed. Saves outputs/corpus_completions.txt.
"""
import os
import re
import sys

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS_FILE = "fineweb_corpus.txt"
OUT = os.path.join("outputs", "corpus_completions.txt")
N_COMPLETIONS = 100
NEW_TOKENS = 20
BEGIN_WORDS = 3                                   # words used as the sentence beginning
COMMON_OPENINGS = ["The", "If", "It", "This", "I", "We", "A", "You", "In"]

ckpt = torch.load(os.path.join("outputs", "model.pt"), weights_only=False)
cfg, vocab = ckpt["config"], ckpt["vocab"]
V, CONTEXT, D_MODEL, N_HEADS, N_LAYERS = cfg["V"], cfg["CONTEXT"], cfg["D_MODEL"], cfg["N_HEADS"], cfg["N_LAYERS"]
to_small = {tok: i for i, tok in enumerate(vocab)}
enc = tiktoken.get_encoding("o200k_base")


class Block(nn.Module):  # same architecture as in train.py
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
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
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        return self.head(self.ln(self.blocks(x)))


# ----------------------------------------------------------------- choose a sentence
corpus = open(CORPUS_FILE, encoding="utf-8").read()
sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", corpus)]
short = [s for s in sentences if 4 <= len(s.split()) <= 15 and s in corpus]


def rank(s):  # common openings first, in the order listed
    first = s.split()[0]
    return COMMON_OPENINGS.index(first) if first in COMMON_OPENINGS else len(COMMON_OPENINGS)


chosen = beginning = begin_ids = None
for s in sorted(short, key=rank):
    b = " ".join(s.split()[:BEGIN_WORDS])
    ids = enc.encode(b)
    if all(t in to_small for t in ids):
        chosen, beginning, begin_ids = s, b, ids
        break
    print(f"Skipped '{b}': tokens not in the vocabulary: "
          f"{[enc.decode([t]) for t in ids if t not in to_small]}")

if chosen is None:
    sys.exit("No suitable sentence found.")

print(f"Chosen sentence: {chosen}")
print(f"Beginning: '{beginning}' = tokens {begin_ids} = {[enc.decode([t]) for t in begin_ids]}\n")

# ----------------------------------------------------------------- generate
model = TinyGPT()
model.load_state_dict(ckpt["model_state"])
model.eval()
torch.manual_seed(0)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"Chosen sentence (appears exactly in the corpus): {chosen}\n")
    f.write(f"Beginning: {beginning}\n\n")
    for n in range(1, N_COMPLETIONS + 1):
        idx = torch.tensor([[to_small[t] for t in begin_ids]])
        with torch.no_grad():
            for _ in range(NEW_TOKENS):
                probs = F.softmax(model(idx[:, -CONTEXT:])[:, -1, :], dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        text = enc.decode([vocab[i] for i in idx[0].tolist()])
        f.write(f"{n:3d}. {text}\n")
        print(f"{n:3d}. {text}")
print(f"\nSaved {OUT}")
