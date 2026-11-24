"""FinBERT in plain NumPy: no PyTorch or transformers needed.

FinBERT (ProsusAI/finbert on Hugging Face) is a BERT model fine-tuned to label financial text
as positive, negative or neutral. This file downloads its weights once (~440 MB, saved in
models/finbert/) and runs the exact same math BERT runs, using NumPy:

    words -> WordPiece tokens -> embeddings -> 12 transformer layers -> classifier -> 3 scores

Why NumPy: PyTorch can't be installed on some computers (for example Intel Macs on Python 3.13+),
and this way the website runs anywhere NumPy runs.
"""
import json
import math
import re
import ssl
import struct
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "finbert"

# config + vocab come from the main repo; the .safetensors copy of the weights lives on a
# Hugging Face pull-request branch of the same repo (with a duplicate repo as a backup)
FILES = {
    "config.json": ["https://huggingface.co/ProsusAI/finbert/resolve/main/config.json"],
    "vocab.txt": ["https://huggingface.co/ProsusAI/finbert/resolve/main/vocab.txt"],
    "model.safetensors": [
        "https://huggingface.co/ProsusAI/finbert/resolve/refs%2Fpr%2F8/model.safetensors",
        "https://huggingface.co/ProsusAI/finbert/resolve/refs%2Fpr%2F7/model.safetensors",
        "https://huggingface.co/Narsil/finbert2/resolve/main/model.safetensors",
    ],
}


# ---------------------------------------------------------------- download

def _ssl_context():
    # python.org installs on Mac often lack root certificates; certifi provides them
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(urls, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "material-weakness/1.0"})
            with urllib.request.urlopen(req, context=_ssl_context(), timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total > 5_000_000:
                        sys.stdout.write(f"\r      {dest.name}: {done / 1e6:5.0f} / {total / 1e6:.0f} MB")
                        sys.stdout.flush()
                if total > 5_000_000:
                    print()
            tmp.rename(dest)
            return
        except Exception as err:  # try the next mirror
            last_err = err
    raise RuntimeError(f"couldn't download {dest.name}: {last_err}")


def download_finbert():
    """Download any FinBERT files that aren't already in models/finbert/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, urls in FILES.items():
        if not (MODEL_DIR / name).exists():
            _download(urls, MODEL_DIR / name)


def is_downloaded():
    return all((MODEL_DIR / n).exists() for n in FILES)


# ---------------------------------------------------------------- safetensors reader

_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64}


def load_safetensors(path):
    """Read a .safetensors file into a dict of NumPy arrays (the format is a JSON header + raw bytes)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__" or info["dtype"] not in _DTYPES:
            continue
        start, end = info["data_offsets"]
        arr = np.frombuffer(data[start:end], dtype=_DTYPES[info["dtype"]]).reshape(info["shape"])
        out[name] = arr.astype(np.float32) if arr.dtype == np.float16 else arr
    return out


# ---------------------------------------------------------------- tokenizer (BERT uncased WordPiece)

def _is_punct(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    def __init__(self, vocab_path):
        self.vocab = {tok.rstrip("\n"): i for i, tok in enumerate(open(vocab_path, encoding="utf-8"))}
        self.unk = self.vocab["[UNK]"]

    def _basic(self, text):
        # clean, lowercase, strip accents, split on whitespace and punctuation
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
                continue
            out.append(f" {ch} " if _is_cjk(cp) else (" " if ch.isspace() else ch))
        words = []
        for word in "".join(out).lower().split():
            word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
            buf = ""
            for ch in word:
                if _is_punct(ch):
                    if buf:
                        words.append(buf)
                    words.append(ch)
                    buf = ""
                else:
                    buf += ch
            if buf:
                words.append(buf)
        return words

    def _wordpiece(self, word):
        if len(word) > 100:
            return [self.unk]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = ("##" if start > 0 else "") + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk]
            ids.append(found)
            start = end
        return ids

    def encode(self, text, max_len=512):
        ids = [i for w in self._basic(text) for i in self._wordpiece(w)]
        ids = ids[: max_len - 2]
        return [self.vocab["[CLS]"]] + ids + [self.vocab["[SEP]"]]


# ---------------------------------------------------------------- the BERT forward pass

def _erf(x):
    # Abramowitz & Stegun 7.1.26 (max error 1.5e-7), since NumPy has no erf
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class FinBERT:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.tok = WordPiece(model_dir / "vocab.txt")
        w = load_safetensors(model_dir / "model.safetensors")
        # older checkpoints name LayerNorm weights gamma/beta; newer ones weight/bias
        self.w = {k.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias"): v
                  for k, v in w.items()}
        self.labels = {int(k): v.lower() for k, v in self.cfg["id2label"].items()}
        self.eps = self.cfg.get("layer_norm_eps", 1e-12)

    def _p(self, name):
        return self.w[name]

    def _linear(self, x, prefix):
        return x @ self._p(prefix + ".weight").T + self._p(prefix + ".bias")

    def logits(self, text):
        ids = np.array(self.tok.encode(text, self.cfg.get("max_position_embeddings", 512)))
        n = len(ids)
        e = "bert.embeddings."
        x = (self._p(e + "word_embeddings.weight")[ids]
             + self._p(e + "position_embeddings.weight")[:n]
             + self._p(e + "token_type_embeddings.weight")[0])
        x = _layer_norm(x, self._p(e + "LayerNorm.weight"), self._p(e + "LayerNorm.bias"), self.eps)

        heads = self.cfg["num_attention_heads"]
        d = x.shape[-1] // heads
        for i in range(self.cfg["num_hidden_layers"]):
            p = f"bert.encoder.layer.{i}."
            # self-attention: every token looks at every other token
            q = self._linear(x, p + "attention.self.query").reshape(n, heads, d).transpose(1, 0, 2)
            k = self._linear(x, p + "attention.self.key").reshape(n, heads, d).transpose(1, 0, 2)
            v = self._linear(x, p + "attention.self.value").reshape(n, heads, d).transpose(1, 0, 2)
            scores = q @ k.transpose(0, 2, 1) / math.sqrt(d)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            attn = scores / scores.sum(-1, keepdims=True)
            ctx = (attn @ v).transpose(1, 0, 2).reshape(n, heads * d)
            x = _layer_norm(x + self._linear(ctx, p + "attention.output.dense"),
                            self._p(p + "attention.output.LayerNorm.weight"),
                            self._p(p + "attention.output.LayerNorm.bias"), self.eps)
            # feed-forward
            h = _gelu(self._linear(x, p + "intermediate.dense"))
            x = _layer_norm(x + self._linear(h, p + "output.dense"),
                            self._p(p + "output.LayerNorm.weight"),
                            self._p(p + "output.LayerNorm.bias"), self.eps)

        pooled = np.tanh(self._linear(x[0], "bert.pooler.dense"))  # the [CLS] token
        return self._linear(pooled, "classifier")

    def predict(self, text):
        z = self.logits(text)
        probs = np.exp(z - z.max())
        probs = probs / probs.sum()
        return {self.labels[i]: float(p) for i, p in enumerate(probs)}
