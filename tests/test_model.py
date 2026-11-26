from minigpt import MiniGPT,AttentionHead

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
def test_para_count():
    """验证参数数量是否为118272"""
    model = MiniGPT(vocab_size=128, max_length=32, channels=64, num_heads=4, num_layers=2)

    total = sum(p.numel() for p in model.parameters())
    assert total == 118272

def test_causal_mask_block_future():
    """改动位置t的输入，只能影响t及t之后，绝不能影响t之前。"""
    torch.manual_seed(26)
    model = MiniGPT(
        vocab_size=128,max_length=32,channels=64,num_heads=4,num_layers=2
    )
    model.eval()

    T = 8
    idx = torch.randint(0,128,(1,T))

    with torch.no_grad():
        base = model(idx)

        for t in range(T):
            changed = idx.clone()
            changed[0,t] = (int(idx[0,t]) + 64) %128

            out = model(changed)

            assert torch.allclose(out[:,:t],base[0,:t],atol=1e-6),(
                f"改了位置{t},位置0...{t-1}变了，mask有漏洞"
            )
            assert not torch.allclose(out[0,t:],base[0,t:],atol=1e-6),(
                f"改了位置{t}，位置{t}及之后完全没变，改动没生效，测试无效"
            )

def test_attention_backward_by_gradcheck():
    """用有限差分核对 attention 的反向。

    边界：只验证"梯度与这个前向自洽、且没有 NaN"。
    不能证明前向的数学是对的——那是第 4 条测试的事。
    """
    torch.manual_seed(0)
    head = AttentionHead(channels=6, head_dim=3, max_length=5).double()

    h = torch.randn(1, 5, 6, dtype=torch.float64, requires_grad=True)

    def attention_output(t):
        out, _ = head(t)
        return out

    ok = gradcheck(attention_output, (h,), eps=1e-6, atol=1e-5, rtol=1e-3)
    assert ok

def test_matches_scaled_dot_product_attention():
    """与官方实现逐元素对比：同时检查缩放因子、mask方向、softmax归一化。
    
    这条能抓到向前的数学错误(例如漏掉1/sqrt(d_head)),而gradcheck抓不到这类错误
    """
    torch.manual_seed(26)
    head = AttentionHead(channels=6,head_dim=3,max_length=5).double()

    h = torch.randn(1,5,6,dtype=torch.float64)

    with torch.no_grad():
        ours,_ = head(h)
        ref = F.scaled_dot_product_attention(
            head.query(h),head.key(h),head.value(h),is_causal=True
        )
    max_diff = (ours - ref).abs().max().item()
    assert torch.allclose(ours,ref,atol=1e-10),(
        f"与官方实现不一致，最大偏差为{max_diff:.3e}"
    )
def test_attention_weights_are_causal_and_normalized():
    """直接检查权重矩阵：未来位置都为零，每行和为1。
    
    与test_matches_scaled_dot_product_attention的分工：那条比最终输出，这条看内部权重，失败时能直接指出是哪个位置错了。    
    """
    torch.manual_seed(26)
    head = AttentionHead(channels=6,head_dim = 3, max_length=5).double()

    h = torch.randn(1,5,6,dtype=torch.float64)

    with torch.no_grad():
        _,w = head(h)

    T = h.shape[1]

    # 1) 未来位置（上三角，不含对角线）的权重必须恰好为 0
    future = torch.triu(torch.ones(T,T,dtype=torch.bool),diagonal=1)
    assert (w[:,future] == 0).all(), "存在位置关注了未来，也就是上三角不全为零"

    # 2) 每行权重之和为 1（softmax 的定义性质）
    row_sums = w.sum(dim = -1)
    max_err = (row_sums - 1).abs().max().item()
    assert max_err < 1e-12, f"行和不为1，最大偏差为{max_err:.3e}"

def test_uniform_mode_has_same_parameter_count():
    """两种模式的参数量必须完全相同
    
    这是消融实验的前提：只有参数量一致，
    结果的差异才能归因于[权重是否可学习]
    
    """
    def build(uniform):
        return MiniGPT(vocab_size=128,max_length=32,channels=64,num_heads=4,num_layers=2,uniform=uniform)
    n_learned = sum(p.numel() for p in build(False).parameters())
    n_uniform = sum(p.numel() for p in build(True).parameters())

    assert n_learned == 118272
    assert n_learned == n_uniform,(f"参数量不一致：学习模式：{n_learned}，均匀模式：{n_uniform}。")

def test_uniform_weights_equal_one_over_t_plus_one():
    """均匀模式下，第t行前t+1个位置权重各为1/(t+1)，其余为0。"""
    torch.manual_seed(26)
    head = AttentionHead(channels=6,head_dim=3,max_length=5,uniform=True).double()

    B,T = 2,5
    h = torch.randn(B,T,6,dtype=torch.float64)

    with torch.no_grad():
        _ , w = head(h)

    assert w.shape == (B,T,T) ,(f"权重形状应为{(B,T,T)},实际为：{tuple(w.shape)}")

    expected = torch.zeros(T,T,dtype=torch.float64)
    for t in range(T):
        expected[t,:t+1] = 1.0/(t+1)

    for b in range(B):
        max_diff = (w[b] - expected).abs().max().item()
        assert torch.allclose(w[b],expected,atol=1e-12),(f"第{b}条样本权重不符，最大偏差为{max_diff:.3e}")