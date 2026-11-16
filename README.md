# mini-gpt

从零实现的字符级语言模型。手写 Transformer，在 [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)（约 1.1 MB）上训练，并与 Bigram、前缀平均两个基线在同一保留测试集上对比。

不依赖 `transformers` 等高层库，全部用 PyTorch 原生算子实现。

## 结果

三个冻结模型，同一保留测试集（语料最后 10%，111,539 个预测位置），同一评分口径：

| 模型 | 参数量 | 如何利用前文 | 测试损失 (nats/字符) | 困惑度 |
|---|---:|---|---:|---:|
| Bigram | 16,384 | 只看当前字符 | 2.5172 | 12.3943 |
| 前缀平均 + MLP | 41,216 | 当前字符 + 因果前缀均值 | 2.4023 | 11.0489 |
| **Mini GPT** | **118,272** | 2 层因果 Transformer | **2.0742** | **7.9581** |

Mini GPT 相对 Bigram 降低 0.4430 nats/字符。

生成样例：

```text
KING DUTIUS:

You hing wost.
```

> ⚠️ **这不是消融实验。** 三个模型的参数量、优化器（Adam / AdamW）和学习率（0.03 / 0.003 / 0.001）都不同，因此上面的差值**不能**归因于「加了 Attention」。控制变量的消融进行中。详见[项目报告](<Mini GPT 项目报告.md>)第 4 节。

## 快速开始

```bash
git clone https://github.com/yangli261/mini-gpt.git
cd mini-gpt

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt

# 冒烟测试：5 步，不写文件，约 10 秒
python stage_05_transformer/08_train_mini_gpt.py --smoke

# 完整训练：1200 步，结果写入 outputs/
python stage_05_transformer/08_train_mini_gpt.py
```

无 GPU 时自动回退到 CPU，可以跑但更慢。有 GPU 时在 GTX 1660 SUPER 上约 24 秒。

## 实现范围

- ✅ 字符级编码 / 80-10-10 连续划分 / 无跨界采样
- ✅ 缩放点积注意力、因果 mask、多头、输出投影
- ✅ 残差 + LayerNorm（Pre-LN）+ 逐位置 MLP
- ✅ 训练循环、固定批次评估、采样生成
- ✅ 权重保存与 `strict=True` 重载
- ✅ 保留测试集统一评测与困惑度
- 🚧 控制变量消融：均匀注意力 vs 学习注意力
- ❌ 子词分词、多卡、混合精度、更大语料

## 目录

| 路径 | 内容 |
|---|---|
| `stage_01` ~ `stage_06` | 逐课推导的教学脚本，按顺序看即可 |
| `data/` | Tiny Shakespeare 语料 |
| `outputs/` | 每次运行的 `config.json`、`history.json`、生成样例（权重文件不入库） |
| `工作日志.md` | 完整推导记录与错题 |

## 复现条件

| 项 | 值 |
|---|---|
| 硬件 | GTX 1660 SUPER（6 GB） |
| PyTorch | 2.7.1+cu126 |
| Python | 3.10 |
| 模型 | `C=64, H=4, L=2, T=32`，118,272 参数 |
| 训练 | batch 32，AdamW，lr 1e-3，1200 步 |
| 种子 | 初始化 42 / 采样 123 / 评估 999 / 生成 2026 |
| 耗时 | 23.6 秒（含评估与采样） |

## 局限

- 语料仅 1.1 MB，模型本质上在记诵这段文本，不能推广到通用语言能力。
- 窗口仅 32 字符，无法利用长程依赖。
- 单次运行、单一配置，没有多种子统计。