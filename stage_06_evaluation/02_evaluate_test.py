"""Compare three frozen character models on the same held-out text."""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 128
BLOCK_SIZE = 32
BATCH_SIZE = 32
CHECKPOINTS = {
    "bigram": ROOT / "outputs" / "bigram" / "model.pt",
    "prefix_mean": ROOT / "outputs" / "no_attention_20261021_203805" / "model.pt",
    "mini_gpt": ROOT / "outputs" / "mini_gpt_20261112_160340" / "model.pt",
}


def import_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_models(device):
    # The original bigram training script runs at import time.
    # Its forward pass is exactly the saved embedding lookup.
    bigram = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
    saved = torch.load(CHECKPOINTS["bigram"], map_location="cpu", weights_only=True)
    bigram.load_state_dict({"weight": saved["model_state"]["table.weight"]}, strict=True)

    prefix_module = import_script(
        "prefix_mean_lesson", ROOT / "stage_05_transformer" / "03_train_no_attention.py"
    )
    saved = torch.load(CHECKPOINTS["prefix_mean"], map_location="cpu", weights_only=True)
    cfg = saved["config"]
    if cfg["encoding"] != "ASCII" or cfg["vocab_size"] != VOCAB_SIZE:
        raise ValueError("Prefix-mean checkpoint uses a different vocabulary")
    prefix = prefix_module.PrefixMeanLM(
        vocab_size=cfg["vocab_size"], channels=cfg["channels"], hidden=cfg["hidden"]
    )
    prefix.load_state_dict(saved["model_state"], strict=True)

    reload_module = import_script(
        "reload_lesson", ROOT / "stage_06_evaluation" / "01_reload_mini_gpt.py"
    )
    mini_gpt, cfg, _ = reload_module.load_model(CHECKPOINTS["mini_gpt"], device)
    if cfg["encoding"] != "ASCII" or cfg["model"]["vocab_size"] != VOCAB_SIZE:
        raise ValueError("Mini GPT checkpoint uses a different vocabulary")
    if cfg["model"]["max_length"] < BLOCK_SIZE:
        raise ValueError("Mini GPT checkpoint does not support the scoring window")

    return {
        "bigram": bigram.to(device).eval(),
        "prefix_mean": prefix.to(device).eval(),
        "mini_gpt": mini_gpt,
    }


def batches(part, block_size, batch_size):
    # Score part[1:] once. Inputs and targets are offset by one position.
    complete, tail = divmod(len(part) - 1, block_size)
    end = complete * block_size
    if complete:
        x = part[:end].reshape(complete, block_size)
        y = part[1:end + 1].reshape(complete, block_size)
        for start in range(0, complete, batch_size):
            yield x[start:start + batch_size], y[start:start + batch_size]
    if tail:
        yield part[end:-1].unsqueeze(0), part[end + 1:].unsqueeze(0)


@torch.no_grad()
def score(model, part, device):
    total_loss, positions = 0.0, 0
    for x, y in batches(part, BLOCK_SIZE, BATCH_SIZE):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if logits.shape != (*y.shape, VOCAB_SIZE):
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        total_loss += F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="sum"
        ).item()
        positions += y.numel()
    if positions != len(part) - 1 or positions == 0 or not math.isfinite(total_loss):
        raise RuntimeError("Invalid target count or loss")
    mean_loss = total_loss / positions
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "positions": positions,
        "loss_nats_per_char": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Score a validation excerpt; save nothing")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = (ROOT / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    if not text.isascii():
        raise ValueError("Expected the original ASCII Tiny Shakespeare corpus")
    n = len(text)
    if args.smoke:
        start = int(0.8 * n)
        stop = min(int(0.9 * n), start + BATCH_SIZE * BLOCK_SIZE + 4)
        split = "validation_smoke"
    else:
        start, stop = int(0.9 * n), n
        split = "test"
    part = torch.tensor([ord(c) for c in text[start:stop]], dtype=torch.long)
    if len(part) < 2:
        raise ValueError("Need at least two characters to score a split")

    models = load_models(device)
    print(f"device={device} split={split} characters={len(part)} block_size={BLOCK_SIZE}", flush=True)
    print(f"{'model':<14} {'params':>10} {'positions':>10} {'loss':>10} {'ppl':>10}", flush=True)
    metrics = {}
    for name, model in models.items():
        m = metrics[name] = score(model, part, device)
        print(
            f"{name:<14} {m['parameters']:>10,} {m['positions']:>10,} "
            f"{m['loss_nats_per_char']:>10.6f} {m['perplexity']:>10.4f}",
            flush=True,
        )

    if args.smoke:
        print("smoke_ok=True (validation only; no output files)")
        return

    result = {
        "split": split,
        "data": "data/tinyshakespeare.txt",
        "encoding": "ASCII",
        "character_range": [start, stop],
        "block_size": BLOCK_SIZE,
        "context_policy": "Reset context at each block; score every next character once, excluding the first character.",
        "checkpoints": {name: path.relative_to(ROOT).as_posix() for name, path in CHECKPOINTS.items()},
        "models": metrics,
    }
    output = ROOT / "outputs" / ("test_evaluation_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
