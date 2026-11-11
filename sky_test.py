"""Load the trained model and complete "The sky is" 100 times.

Run from the 4. Build-A-LLM folder (with the .venv active):
    python3 sky_test.py

Saves outputs/sky_completions.txt. If "The sky is" contains tokens the model never saw
in training, it says which ones and stops: a model can only produce tokens from its vocabulary.
"""
import os
import sys

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

PROMPT = "The sky is"
N_COMPLETIONS = 100
NEW_TOKENS = 20
OUT = os.path.join("outputs", "sky_completions.txt")

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


prompt_ids = enc.encode(PROMPT)
print(f"'{PROMPT}' = tokens {prompt_ids} = {[enc.decode([t]) for t in prompt_ids]}")
missing = [t for t in prompt_ids if t not in to_small]
if missing:
    print(f"FAILED: the model has never seen these tokens: {[enc.decode([t]) for t in missing]}")
    print("It cannot read a beginning that contains them. Use the corpus-sentence script instead.")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {PROMPT}\nFailed: tokens not in the training vocabulary: "
                f"{[enc.decode([t]) for t in missing]}\n")
    sys.exit(0)

model = TinyGPT()
model.load_state_dict(ckpt["model_state"])
model.eval()
torch.manual_seed(0)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"Prompt: {PROMPT}\n\n")
    for n in range(1, N_COMPLETIONS + 1):
        idx = torch.tensor([[to_small[t] for t in prompt_ids]])
        with torch.no_grad():
            for _ in range(NEW_TOKENS):
                probs = F.softmax(model(idx[:, -CONTEXT:])[:, -1, :], dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        text = enc.decode([vocab[i] for i in idx[0].tolist()])
        f.write(f"{n:3d}. {text}\n")
        print(f"{n:3d}. {text}")
print(f"\nSaved {OUT}")
