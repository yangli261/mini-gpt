import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from minigpt import MiniGPT

@torch.no_grad()
def generate(model,idx,new_tokens):
    for _ in range(new_tokens):
        window = idx[:,-model.max_length:]
        next_logits = model(window)[:,-1,:]
        next_id = torch.multinomial(next_logits.softmax(dim = -1),num_samples=1)
        idx = torch.cat((idx,next_id),dim = 1)
    return idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",choices=["learned","uniform"],default="learned")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps , eval-every and learning rate must be positive and finite")

    steps = 5 if args.smoke else args.steps

    #模型初始化与训练采样都由seed决定；评估种子固定，保证两组比的是同一批数据
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = (root/"data"/"tinyshakespeare.txt").read_text(encoding="utf-8")
    if not text.isascii():
        raise ValueError("Expected the original ASCII Tiny Shakespeare corpus")

    data = torch.tensor([ord(c) for c in text],dtype=torch.long)
    n = len(data)
    splits = {
        "train": data[:int(0.8*n)],
        "val": data[int(0.8*n):int(0.9*n)],
        "test": data[int(0.9*n):],
    }

    batch_size = 32
    model_config = {
        "vocab_size": 128,
        "max_length": 32,
        "channels": 64,
        "num_heads": 4,
        "num_layers": 2,
        "uniform": args.variant == "uniform",
    }
    V,T = model_config["vocab_size"],model_config["max_length"]
    if any(len(part) <= T for part in splits.values()):
        raise ValueError("Every split must have at least T+1 characters")

    def get_batch(split,rng):
        part = splits[split]
        starts = torch.randint(len(part)-T,(batch_size,),generator=rng).tolist()
        x = torch.stack([part[i:i+T] for i in starts])
        y = torch.stack([part[i+1:i+1+T] for i in starts])
        return x.to(device),y.to(device)

    model = MiniGPT(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr)
    train_rng = torch.Generator().manual_seed(args.seed + 1)

    def batch_loss(x,y):
        logits = model(x)
        return F.cross_entropy(logits.reshape(-1,V),y.reshape(-1))

    @torch.no_grad()
    def evaluate():
        was_training = model.training
        model.eval()
        metrics = {}
        for split in ("train","val"):
            rng = torch.Generator().manual_seed(999)
            losses = [batch_loss(*get_batch(split,rng)).item() for _ in range(10)]

            metrics[split] = sum(losses)/len(losses)
        model.train(was_training)
        return metrics

    count = sum(p.numel() for p in model.parameters())
    print(
        f"variant = {args.variant} seed = {args.seed} parameters = {count:,}",
        flush=True
    )

    start = time.perf_counter()
    history = []
    model.train()
    for step in range(steps + 1):
        if step % args.eval_every == 0 or step == steps:
            metrics = evaluate()
            if not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError("Non-finite evaluation loss")
            history.append({"step":step,**metrics})
            print(
                f"step={step:4d} train={metrics['train']:.4f} val = {metrics['val']:.4f}",
                flush=True,
            )
            if step == steps:
                break
        x,y = get_batch("train",train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(x,y)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start

    if args.smoke:
        print(f"smoke_ok=True elapsed={elapsed:.1f}s no output files written",flush=True)
        return


    model.eval()
    with torch.no_grad():
        torch.manual_seed(2026)
        prompt = torch.tensor([[ord("\n")]],device=device)
        tokens = generate(model,prompt,new_tokens=300)
        sample = "".join(chr(i) for i in tokens[0].cpu().tolist())

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = root/"outputs"/f"ablation_{args.variant}_seed{args.seed}_{stamp}"
    out.mkdir(parents=True,exist_ok=False)

    config = {
        "model": model_config,
        "variant": args.variant,
        "seed": args.seed,
        "batch_size": batch_size,
        "steps": steps,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "eval_seed": 999,
        "eval_batches_per_split": 10,
        "sample_seed": 2026,
        "parameters": count,
        "seconds": elapsed,
        "split_lengths": {k: len(v) for k,v in splits.items()},
    }

    torch.save({"model_state":model.state_dict(),"config": config},out / "model.pt")
    (out / "history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (out / "config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    (out / "sample.txt").write_text(sample, encoding="utf-8")
    print(f"elapsed={elapsed:.1f}s output={out}",flush=True)


if __name__ == "__main__":
    main()