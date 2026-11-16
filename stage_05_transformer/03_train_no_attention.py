"""Causal prefix-average language model, without attention or recurrence.

Run from project root:
  .\.venv\Scripts\python.exe .\stage_05_transformer\03_train_no_attention.py
"""
from pathlib import Path
import argparse
import json
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class PrefixMeanLM(nn.Module):
    def __init__(self, vocab_size=128, channels=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        # Keep the current character separate from the pooled prefix.
        self.head = nn.Sequential(
            nn.Linear(2 * channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, vocab_size),
        )
        # Uniform initial probabilities, as in the earlier bigram experiment.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, idx):
        h = self.embedding(idx)  # [B,T,C]
        lengths = torch.arange(
            1, idx.shape[1] + 1, device=h.device, dtype=h.dtype
        ).view(1, -1, 1)
        prefix_mean = h.cumsum(dim=1) / lengths
        features = torch.cat((h, prefix_mean), dim=-1)
        return self.head(features)  # [B,T,V]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        parser.error("steps and learning rate must be positive")

    torch.manual_seed(42)
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = (root / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    if not text.isascii():
        raise ValueError("This experiment uses the fixed 128-character ASCII vocabulary")
    vocab_size, batch_size, block_size = 128, 32, 32
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    n = len(data)
    splits = {
        "train": data[:int(.8 * n)],
        "val": data[int(.8 * n):int(.9 * n)],
        "test": data[int(.9 * n):],
    }
    if any(len(part) <= block_size for part in splits.values()):
        raise ValueError("Each split must contain more than block_size characters")
    train_rng = torch.Generator().manual_seed(123)

    def get_batch(split, rng):
        part = splits[split]
        starts = torch.randint(
            len(part) - block_size, (batch_size,), generator=rng
        ).tolist()
        x = torch.stack([part[i:i + block_size] for i in starts])
        y = torch.stack([part[i + 1:i + block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    def loss_for(model, x, y):
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

    @torch.no_grad()
    def evaluate(model):
        was_training = model.training
        model.eval()
        results = {}
        for split in ("train", "val"):
            rng = torch.Generator().manual_seed(999)
            losses = []
            for _ in range(10):
                x, y = get_batch(split, rng)
                losses.append(loss_for(model, x, y).item())
            results[split] = sum(losses) / len(losses)
        model.train(was_training)
        return results

    model = PrefixMeanLM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} parameters={param_count} steps={args.steps} lr={args.lr}", flush=True)
    history = []
    start_time = time.perf_counter()
    interval = max(1, args.steps // 4)
    for step in range(args.steps + 1):
        if step % interval == 0 or step == args.steps:
            metrics = evaluate(model)
            history.append({"step": step, **metrics})
            print(f"step={step:4d} train={metrics['train']:.4f} val={metrics['val']:.4f}", flush=True)
        if step == args.steps:
            break
        x, y = get_batch("train", train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        optimizer.step()

    # Test causality after training so nonzero learned weights are exercised.
    model.eval()
    with torch.no_grad():
        probe = torch.tensor([[ord(c) for c in "hello"]], device=device)
        changed = probe.clone()
        changed[:, -1] = ord("x")
        before = model(probe)
        after = model(changed)
        assert before.shape == (1, 5, vocab_size)
        assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
        batch_probe = torch.cat((probe, changed), dim=0)
        assert torch.allclose(model(batch_probe)[0], before[0], atol=1e-5)
        print("Causality and sample isolation checks passed", flush=True)

        # Identical sampling policy to the bigram lesson: temperature=1, 300 new characters.
        torch.manual_seed(2026)
        tokens = torch.tensor([[ord("\n")]], dtype=torch.long, device=device)
        for _ in range(300):
            logits = model(tokens[:, -block_size:])[:, -1, :]
            next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
            tokens = torch.cat((tokens, next_id), dim=1)
        sample = "".join(chr(i) for i in tokens[0].cpu().tolist())

    comparison = {"prefix_mean": history[-1]}
    bigram_path = root / "outputs" / "bigram" / "model.pt"
    if bigram_path.exists():
        saved = torch.load(bigram_path, map_location=device, weights_only=True)
        bigram = nn.Embedding(vocab_size, vocab_size).to(device)
        bigram.load_state_dict({"weight": saved["model_state"]["table.weight"]})
        comparison["bigram"] = evaluate(bigram)
        print("Same fixed evaluation batches:", json.dumps(comparison), flush=True)

    elapsed = time.perf_counter() - start_time
    config = {
        "architecture": "current_embedding_plus_causal_prefix_mean_MLP",
        "attention": False, "positional_encoding": False,
        "vocab_size": vocab_size, "encoding": "ASCII", "channels": 64, "hidden": 128,
        "batch_size": batch_size, "block_size": block_size,
        "steps": args.steps, "lr": args.lr, "seed": 42,
        "parameters": param_count, "seconds": elapsed,
        "split_lengths": {k: len(v) for k, v in splits.items()},
        "comparison_caveat": "Different parameter counts, learning rates and compute; not an isolated attention ablation.",
    }
    out = root / "outputs" / ("no_attention_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"model_state": model.state_dict(), "config": config}, out / "model.pt")
    for name, value in (("history", history), ("comparison", comparison), ("config", config)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    (out / "sample.txt").write_text(sample, encoding="utf-8")
    print("sample:", repr(sample), flush=True)
    print(f"elapsed={elapsed:.1f}s output={out}", flush=True)


if __name__ == "__main__":
    main()
