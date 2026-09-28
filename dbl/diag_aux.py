"""诊断: 辅助损失为什么学不动。

三步定位:
  1. 单独优化辅助损失，router 能否学会可分数据？(排除容量/学习率)
  2. 梯度是否真的回流到 router_big？(排除断链)
  3. aux 与 LM 对 router_big 的梯度量级对比？(定位是否被压制)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dbl.config import Config
from dbl.moe import TrainMoE

DIM = 32
N_EXPERTS, TOP_K, RANK = 8, 2, 4
DEV = "cuda:0"


class MLP(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim, bias=False)
        self.down = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def make(seed=0):
    torch.manual_seed(seed)
    cfg = Config(
        model_id="dummy", num_experts=N_EXPERTS, top_k=TOP_K, lora_rank=RANK,
        lora_alpha=RANK, gamma=0.3, group_sizes=(2, 2, 4),
        device=DEV, dtype="float32",
    )
    mods = []
    for _ in range(4):
        m = TrainMoE(MLP().to(DEV), DIM, cfg, device=DEV, dtype=torch.float32)
        with torch.no_grad():
            m.lora_B.normal_(0, 0.05)
        m.collect_stats = True
        mods.append(m)
    return cfg, mods


def syn_data(n=64, seed=1):
    """可分数据：domain 0 第 0 维为正，domain 1 为负。"""
    g = torch.Generator().manual_seed(seed)
    y_cpu = torch.randint(0, 2, (n,), generator=g)
    x_cpu = torch.randn(n, DIM, generator=g)
    x_cpu[:, 0] += (y_cpu == 0).float() * 3.0 - (y_cpu == 1).float() * 3.0
    return x_cpu.unsqueeze(0).to(DEV), y_cpu.to(DEV)


def step1():
    print("=" * 66)
    print("步骤 1: 只优化辅助损失 —— router 能否学会可分数据？")
    print("=" * 66)
    cfg, mods = make()
    x, y = syn_data()
    crit = nn.CrossEntropyLoss()
    params = [p for m in mods for n, p in m.named_parameters()
              if n.startswith(("router_", "lora_"))]
    opt = torch.optim.AdamW(params, lr=1e-2)

    for _ in range(300):
        opt.zero_grad()
        total = 0.0
        for m in mods:
            m(x)                                   # 填充 last_stats
            st = m.last_stats
            total = total + crit(m.router_big(x)[0], y)
            total = total + crit(
                cfg.groups.pool_to_groups(m.router_little(x))[0], y % 3
            )
            total = total + 0.01 * m.num_experts * torch.sum(
                st.topk_weights.mean(0) * st.little_probs.mean(0)
            )
        total.backward()
        opt.step()

    with torch.no_grad():
        m0 = mods[0]
        loss = crit(m0.router_big(x)[0], y)
        acc = (m0.router_big(x)[0].argmax(-1) == y).float().mean()
    print(f"  300 步后  big_router loss = {loss:.4f}   准确率 = {acc:.2%}")
    print("  随机猜测   loss = 0.6931   准确率 = 50%")
    verdict = "能学会" if loss < 0.3 else "学不动"
    print(f"  -> {verdict}\n")


def step2():
    print("=" * 66)
    print("步骤 2: 梯度是否回流到 router_big？")
    print("=" * 66)
    cfg, mods = make()
    x, y = syn_data(n=8)
    crit = nn.CrossEntropyLoss()
    for m in mods:
        m(x)                              # 填充 last_stats

    for m in mods:
        m.zero_grad()
    total = sum(crit(m.last_stats.big_logits[0], y) for m in mods)
    total.backward()
    for i, m in enumerate(mods):
        g = m.router_big.weight.grad
        print(f"  层{i}  aux only: grad |max| = "
              f"{0.0 if g is None else float(g.abs().max()):.3e}")

    for m in mods:
        m.zero_grad()
    sum(m(x).sum() for m in mods).backward()
    for i, m in enumerate(mods):
        g = m.router_big.weight.grad
        print(f"  层{i}  LM  only: grad |max| = "
              f"{0.0 if g is None else float(g.abs().max()):.3e}")
    print()


def step3():
    print("=" * 66)
    print("步骤 3: aux 与 LM 对 router_big 的梯度量级")
    print("=" * 66)
    cfg, mods = make()
    x, y = syn_data(n=8)
    crit = nn.CrossEntropyLoss()
    for m in mods:
        m(x)

    def gmax(build):
        for m in mods:
            m.zero_grad()
        build()                       # 每次重新前向，避免复用已释放的图
        return max(float(m.router_big.weight.grad.abs().max()) for m in mods)

    crit = nn.CrossEntropyLoss()

    def build_lm():
        sum(m(x).sum() for m in mods).backward()

    def build_aux():
        for m in mods:
            m(x)
        sum(crit(m.last_stats.big_logits[0], y) for m in mods).backward()

    g_lm = gmax(build_lm)
    g_aux = gmax(build_aux)

    print(f"  LM  loss 梯度 |max| = {g_lm:.3e}")
    print(f"  aux loss 梯度 |max| = {g_aux:.3e}")
    print(f"  aux/LM 比值         = {g_aux / max(g_lm, 1e-12):.4f}")
    print()
    w = 0.1
    print(f"  按 router_aux_weight={w} 加权后：")
    print(f"    有效 aux 梯度 = {g_aux * w:.3e}  vs  LM = {g_lm:.3e}")
    print(f"    aux / LM      = {g_aux * w / max(g_lm, 1e-12):.4f}")
    print()
    print("  LM loss 会不会主动把 router 推向某个方向？")
    print("  下面测：在只有 LM 梯度时训练 300 步，看 router 的域区分度")
    cfg2, mods2 = make()
    x2, y2 = syn_data()
    opt = torch.optim.AdamW(
        [p for m in mods2 for p in m.router_big.parameters()], lr=1e-2
    )
    for _ in range(300):
        opt.zero_grad()
        sum(m(x2).sum() for m in mods2).backward()
        opt.step()
    with torch.no_grad():
        d0 = mods2[0].router_big(x2)[0, y2 == 0].mean(0)
        d1 = mods2[0].router_big(x2)[0, y2 == 1].mean(0)
        sep = float((d0 - d1).norm())
    print(f"    LM-only 训练后两类 logits 的分离度 |mean0-mean1| = {sep:.4f}")
    print("    初始随机初始化时约为 O(0.1~1)，接近 0 说明没学出方向")


if __name__ == "__main__":
    step1()
    step2()
    step3()
