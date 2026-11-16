# 从零训练Mini GPT

项目根目录：`D:\ChatGPT\ailearning\mini-gpt`。最后更新：2026-11-14。

## 当前进度

阶段一至四的基础课程已完成。阶段五中，前缀平均、无Attention模型训练、单头Attention梯度与位置Embedding实验已验证；多头Attention的形状、因果约束和梯度实验也已通过；两层Pre-LN Block的形状、因果性和梯度已由用户运行验证；字符级Mini GPT已在GPU上正式训练1200步，固定验证损失2.0657，权重重新加载和生成复现均通过。正式保留测试已完成：Mini GPT测试损失2.074189，困惑度7.9581；同口径下优于Bigram和前缀平均MLP。六个阶段均已完成。[最终项目报告](Mini GPT 项目报告.md)汇总设计、复现与结论边界。原始结果见[测试指标](outputs/test_evaluation_20261114_144839/test_metrics.json)。

完整进度表、教学代码、运行结果与错题见[工作日志](工作日志.md)。

## 目录

- stage_01_environment：环境检查。
- stage_02_pytorch：张量、自动求导、训练循环与模块。
- stage_03_data：字符编码、数据划分与采样。
- stage_04_baseline：Embedding、交叉熵与bigram训练。
- stage_05_transformer：前缀平均、Attention、位置表示及后续模块。
- stage_06_evaluation：权重加载与最终测试集对比。
- outputs：bigram、无Attention模型和[本次Mini GPT](outputs/mini_gpt_20261112_160340)的权重、记录和生成文本。

## 运行

从项目根目录使用共享虚拟环境：

```powershell
.\.venv\Scripts\python.exe .\stage_05_transformer\08_train_mini_gpt.py --smoke
.\.venv\Scripts\python.exe -u .\stage_05_transformer\08_train_mini_gpt.py
```

07_transformer_block.py已由用户运行通过；08_train_mini_gpt.py支持 --smoke 短测（不保存产物），无参数时训练1200步并保存权重和样例。GPU为GTX 1660 SUPER（6 GB），PyTorch为2.7.1+cu126，已验证可用。
