"""Character-level Mini GPT using the preceding lesson's Transformer Block."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, channels, head_dim, max_length):
        super().__init__()

        self.query = nn.Linear(channels, head_dim, bias=False)
        self.key = nn.Linear(channels, head_dim, bias=False)
        self.value = nn.Linear(channels, head_dim, bias=False)

        self.register_buffer(
            "allowed",
            torch.tril(
                torch.ones(max_length, max_length, dtype=torch.bool)
            ),
        )

    def forward(self, h):
        T = h.shape[1]
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(
            ~self.allowed[:T, :T], float("-inf")
        )

        weights = scores.softmax(dim=-1)
        return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, channels=8, num_heads=2, max_length=4):
        super().__init__()
        assert channels % num_heads == 0
        head_dim = channels // num_heads

        self.heads = nn.ModuleList([
            AttentionHead(channels, head_dim, max_length)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_dim, channels
        )

    def forward(self, h):
        results = [head(h) for head in self.heads]

        joined = torch.cat(
            [value for value, _ in results], dim=-1
        )
        weights = torch.stack(
            [weight for _, weight in results], dim=1
        )

        return self.projection(joined), weights

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, max_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = MultiHeadAttention(channels, num_heads, max_length)
        self.ln2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, h):
        update, _ = self.attn(self.ln1(h))
        h = h + update
        h = h + self.ffn(self.ln2(h))
        return h


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_length, channels, num_heads, num_layers):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, channels)
        self.position_embedding = nn.Embedding(max_length, channels)
        self.blocks = nn.Sequential(*(
            TransformerBlock(channels, num_heads, max_length)
            for _ in range(num_layers)
        ))
        self.final_norm = nn.LayerNorm(channels)
        self.lm_head = nn.Linear(channels, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        if not 1 <= T <= self.max_length:
            raise ValueError("Input length must be in [1, max_length]")
        positions = torch.arange(T, device=idx.device)
        h = self.token_embedding(idx) + self.position_embedding(positions)
        h = self.blocks(h)
        return self.lm_head(self.final_norm(h))


@torch.no_grad()
def generate(model, idx, new_tokens):
    for _ in range(new_tokens):
        # Keep the latest window; the full generated history stays in idx.
        window = idx[:, -model.max_length:]
        next_logits = model(window)[:, -1, :]
        next_id = torch.multinomial(next_logits.softmax(dim=-1), num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--smoke", action="store_true", help="5 updates, no output files")
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps and learning rate must be positive and finite")

    steps = 5 if args.smoke else args.steps
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same encoding, split and sampling as the earlier baselines.
    text = (root / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    if not text.isascii():
        raise ValueError("Expected the original ASCII Tiny Shakespeare corpus")
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    n = len(data)
    splits = {
        "train": data[:int(0.8 * n)],
        "val": data[int(0.8 * n):int(0.9 * n)],
        "test": data[int(0.9 * n):],
    }
    batch_size = 32
    model_config = {
        "vocab_size": 128,
        "max_length": 32,
        "channels": 64,
        "num_heads": 4,
        "num_layers": 2,
    }
    V, T = model_config["vocab_size"], model_config["max_length"]
    if any(len(part) <= T for part in splits.values()):
        raise ValueError("Every split must have at least T + 1 characters")

    def get_batch(split, rng):
        part = splits[split]
        starts = torch.randint(len(part) - T, (batch_size,), generator=rng).tolist()
        x = torch.stack([part[i:i + T] for i in starts])
        y = torch.stack([part[i + 1:i + T + 1] for i in starts])
        return x.to(device), y.to(device)

    model = MiniGPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = torch.Generator().manual_seed(123)

    def batch_loss(x, y):
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    @torch.no_grad()
    def evaluate():
        was_training = model.training
        model.eval()
        metrics = {}
        for split in ("train", "val"):
            # Same seed and ten batches per split as the baseline scripts.
            rng = torch.Generator().manual_seed(999)
            losses = [batch_loss(*get_batch(split, rng)).item() for _ in range(10)]
            metrics[split] = sum(losses) / len(losses)
        model.train(was_training)
        return metrics

    count = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={count:,} steps={steps} lr={args.lr}", flush=True)
    start = time.perf_counter()
    history = []
    model.train()
    for step in range(steps + 1):
        if step % 300 == 0 or step == steps:
            metrics = evaluate()
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError("Non-finite evaluation loss")
            history.append({"step": step, **metrics})
            print(f"step={step:4d} train={metrics['train']:.4f} val={metrics['val']:.4f}", flush=True)
        if step == steps:
            break

        x, y = get_batch("train", train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(x, y)
        loss.backward()
        if step == 0:
            # Demonstrate that the loss reaches the first block, then verify step().
            weight = model.blocks[0].attn.heads[0].query.weight
            grad_norm = weight.grad.norm().item()
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError("Expected a finite nonzero gradient in the first block")
            before_step = weight.detach().clone()

        optimizer.step()

        if step == 0:
            if torch.equal(weight.detach(), before_step):
                raise RuntimeError("First optimizer step did not change Query weights")
            print(f"first_batch={tuple(x.shape)} query_grad={grad_norm:.6f} query_updated=True", flush=True)

    model.eval()
    with torch.no_grad():
        probe = torch.tensor([[ord(c) for c in "hello"]], device=device)
        changed = probe.clone()
        changed[:, -1] = ord("x")
        logits = model(probe)
        altered = model(changed)
        if logits.shape != (1, 5, V):
            raise RuntimeError("Unexpected output shape")
        if not torch.allclose(logits[:, :-1], altered[:, :-1], atol=1e-6):
            raise RuntimeError("Earlier predictions changed when the last token changed")
        print(f"probe_logits={tuple(logits.shape)} causal_check=True", flush=True)

        torch.manual_seed(2026)
        prompt = torch.tensor([[ord("\n")]], device=device)
        tokens = generate(model, prompt, new_tokens=40 if args.smoke else 300)
        sample = "".join(chr(i) for i in tokens[0].cpu().tolist())

    print("sample:", repr(sample), flush=True)
    elapsed = time.perf_counter() - start
    if args.smoke:
        print(f"smoke_ok=True elapsed={elapsed:.1f}s no output files written", flush=True)
        return

    out = root / "outputs" / ("mini_gpt_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    config = {
        "model": model_config,
        "encoding": "ASCII",
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "model_seed": 42,
        "train_seed": 123,
        "eval_seed": 999,
        "sample_seed": 2026,
        "eval_batches_per_split": 10,
        "split_lengths": {name: len(part) for name, part in splits.items()},
        "parameters": count,
        "seconds": elapsed,
    }
    torch.save({"model_state": model.state_dict(), "config": config}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "sample.txt").write_text(sample, encoding="utf-8")
    print(f"elapsed={elapsed:.1f}s output={out}", flush=True)


if __name__ == "__main__":
    main()
