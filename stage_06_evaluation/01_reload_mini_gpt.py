"""Load the saved Mini GPT and reproduce validation loss and generation."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "stage_05_transformer" / "08_train_mini_gpt.py"


def load_model(checkpoint_path, device):
    # The file name starts with 08, so load the module by path.
    spec = importlib.util.spec_from_file_location("mini_gpt_lesson", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model definition: {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = module.MiniGPT(**config["model"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, config, module.generate


@torch.no_grad()
def fixed_validation_loss(model, config, device):
    text = (ROOT / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    n = len(data)
    val = data[int(0.8 * n):int(0.9 * n)]
    T = config["model"]["max_length"]
    V = config["model"]["vocab_size"]
    B = config["batch_size"]
    rng = torch.Generator().manual_seed(config["eval_seed"])
    losses = []
    for _ in range(config["eval_batches_per_split"]):
        starts = torch.randint(len(val) - T, (B,), generator=rng).tolist()
        x = torch.stack([val[i:i + T] for i in starts]).to(device)
        y = torch.stack([val[i + 1:i + T + 1] for i in starts]).to(device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a saved Mini GPT model.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, generate = load_model(args.checkpoint, device)
    print(f"device={device} strict_load=True")

    val_loss = fixed_validation_loss(model, config, device)
    print(f"fixed_val_loss={val_loss:.8f}")
    history_path = args.checkpoint.parent / "history.json"
    if history_path.is_file():
        recorded = json.loads(history_path.read_text(encoding="utf-8"))[-1]["val"]
        print(f"recorded_val_loss={recorded:.8f} matches={math.isclose(val_loss, recorded, abs_tol=1e-5)}")

    torch.manual_seed(config["sample_seed"])
    prompt = torch.tensor([[ord("\n")]], device=device)
    tokens = generate(model, prompt, new_tokens=300)
    sample = "".join(chr(i) for i in tokens[0].cpu().tolist())
    print(f"generated_chars={tokens.shape[1] - prompt.shape[1]}")
    sample_path = args.checkpoint.parent / "sample.txt"
    if sample_path.is_file():
        # Writing text on Windows can translate LF to CRLF.
        saved = sample_path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        print(f"sample_matches_saved={sample == saved}")
    print("sample:", repr(sample))


if __name__ == "__main__":
    main()
