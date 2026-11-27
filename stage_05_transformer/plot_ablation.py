"""把6次消融运行结果聚合成一张对比图。
运行：
    .\.venv\Scripts\python.exe .\stage_05_transformer\plot_ablation.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg") # 无界面后端，直接存文件，不弹窗
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"

# Windows 上必须显式指定中文字体，否则中文全是方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

LABELS = {
    "learned": "学习注意力(softmax)",
    "uniform": "均匀注意力(固定 1/(t+1))",
}
def collect(variant):
    """读入某个变体的全部种子。"""
    paths = sorted(OUTPUT.glob(f"ablation_{variant}_seed*/history.json"))
    if not paths:
        raise SystemExit(f"找不到ablation_{variant}_seed*/history.json")
    return [json.loads(p.read_text(encoding = "utf-8")) for p in paths]


def summarize(runs):
    """多个种子压成 均值 与 样本标准差。"""
    steps = np.array([point["step"] for point in runs[0]])
    table = np.array([[point["val"] for point in run] for run in runs])
    return steps, table.mean(axis=0), table.std(axis=0,ddof=1)


def main():
    fig, ax = plt.subplots(figsize=(6.5,4.2))

    for variant,label in LABELS.items():
        runs = collect(variant)
        steps, mean, std = summarize(runs)
        ax.plot(
            steps, mean, marker = "o", markersize = 4, label=f"{label}({len(runs)}个种子)",
        )
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.annotate(
            f"{mean[-1]:.4f} ± {std[-1]:.4f}",
            xy=(steps[-1], mean[-1]),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("训练步数")
    ax.set_ylabel("固定验证损失（nats/字符）")
    ax.set_title("注意力权重是学习的还是固定的：验证损失对比")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = OUTPUT / "ablation_curve.png"
    fig.savefig(out, dpi=160,bbox_inches="tight")
    print(f"已保存到 {out}")

if __name__ == "__main__":
    main()

