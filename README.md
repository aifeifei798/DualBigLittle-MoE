# DualBigLittle-MoE

**在消费级显卡上用「大核 + 小核」分层结构做文理双域 MoE**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Consumer_GPU-green.svg)](https://github.com/)

---

## 1. 这个项目解决什么

标准 MoE 有两个绕不开的问题：

1. **领域偏斜**——模型在写作、常识上表现好，在代码、数学上明显退化
2. **显存墙**——MoE 的 FFN 专家体积大，扩展即显存爆炸，只能上多卡集群

DualBigLittle-MoE 的思路是借鉴移动端三簇 CPU（大核 / 小核 / 能效核），把模型容量拆到三个物理位置：

| 层 | 位置 | 内容 | 作用 |
| :--- | :--- | :--- | :--- |
| **Tier 1** 文科锚核 | GPU 显存 | 原版 MLP，**完全冻结** | 语言底座，保证不退化 |
| **Tier 2** 理科孪生核 | GPU 显存 | 克隆 MLP，低学习率微调 | 代码、数学、逻辑 |
| **Tier 3** 微专家池 | 主机 pinned RAM | 896 个 rank-16 LoRA，**56 MiB** | 细粒度专家，按需流式搬运 |

```
                              Token 激活 x
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            ┌─────────────────┐         ┌─────────────────┐
            │ Tier 1 文科锚核  │         │ Tier 2 理科孪生核│
            │  原版 MLP · 冻结 │         │  克隆 MLP · 微调 │
            │  常驻 VRAM      │         │  常驻 VRAM      │
            └────────┬────────┘         └────────┬────────┘
                     └────────────┬──────────────┘
                                  ▼
                        y_big = w_arts·FFN_arts
                              + w_sci ·FFN_sci
                                  │
                                  ▼
            ┌────────────────────────────────────────┐
            │ Tier 3 微专家池（主机 pinned RAM）      │
            │ 896 个 rank-16 LoRA，每层选 Top-8      │
            │ 经 PCIe DMA 流式搬入 compute buffer     │
            │ #0-7 代码 │ #8-15 数学 │ #16-31 写作    │
            └───────────────────┬────────────────────┘
                                ▼
              y = y_big + 0.3 · Σ wᵢ · LoRAᵢ(x)
```

---

## 2. 实测数据

全部数据在 **RTX 5090 D** 上测得，可复现（见 §5）。

### 资源占用

| 项目 | 数值 |
| :--- | :--- |
| Qwen3-0.6B 基座 | 1136.9 MiB (bf16) |
| Tier-2 理科大核增量 | **+504.0 MiB** |
| 路由器（28 层 × 2） | +3.7 MiB |
| **常驻 VRAM 合计** | **1671.3 MiB** |
| **生成峰值 VRAM** | **1719.8 MiB** |
| **Tier-3 pinned 主机内存** | **56.0 MiB**（896 × 64 KiB） |
| 训练峰值显存 | 12.73 GB |

关键点：896 个微专家**完全不占 VRAM**。

### 吞吐

| 任务 | 速度 |
| :--- | :--- |
| 训练 | 5.11 分钟跑完 8000 条（26.1 samples/s，500 步） |
| 推理（代码） | 26.0 tok/s |
| 推理（数学） | 25.8 tok/s |
| 推理（散文） | 25.9 tok/s |

### 路由分化

逐层统计 Code 语料 vs Arts 语料的理科核权重：

```
   层     Code     Arts       差值
   0    76.8%    41.1%    +35.7%
   8    93.1%    18.3%    +74.8%
  16    94.8%     5.2%    +89.5%
  27    95.9%     0.7%    +95.2%    <-- 末层判别力最强
```

末层 Code 走理科核 95.8%，Arts 只走 4.4%。

### 困惑度（vs 原生 Qwen3-0.6B）

| 领域 | baseline | 双大核 | 变化 |
| :--- | :--- | :--- | :--- |
| Code | 10.9 | **4.9** | ↓ 55% |
| Math | 7.7 | **3.8** | ↓ 51% |
| Arts | 35.0 | **21.2** | ↓ 40% |

三个领域全部改善，理科改善更明显，文科没有因为冻结 Tier-1 而停滞。

### 专家健康度

| 指标 | 数值 |
| :--- | :--- |
| 获得非零权重的专家 | **896 / 896** |
| 完全未被调用的专家 | **0 / 32**（每层） |
| 40 次相同前向的数值抖动 | **0.00** |

---

## 3. 关键实现

### 3.1 微专家用堆叠权重，不是 ModuleList

32 个 LoRA 存成两块 `(E, r, D)` 张量而非 32 个 `nn.Module`：

```python
self.lora_A = nn.Parameter(torch.empty(num_experts, rank, hidden_dim, ...))
self.lora_B = nn.Parameter(torch.empty(num_experts, rank, hidden_dim, ...))

# 前向：两次大 matmul 完成全部 32 个专家
h = torch.matmul(x.reshape(n, dim), a_flat.t())   # (N, E*r)
h = h * topk_mask_expanded                        # top-k 归一化权重
y = torch.matmul(h, b_flat)                       # (N, D)
```

这样做的收益：

- **梯度覆盖全部专家**。早期版本用 `ModuleList` + 单专家索引，8000 条语料的 label 只有 0/8/16 三个值，导致每层只有 3 个专家拿到梯度，其余 29 个因 `lora_B` 零初始化且无梯度更新而**输出恒为 0**——名义 896 个专家，实际 93 个能工作。
- **不物化 `(N, E, D)` 激活**。稠密算完再按 top-k 掩码，显存可控。
- **与推理路径数学等价**。训练用掩码、推理用 Top-8 搬运，公式完全一致（bf16 下相差 0.08 ULP）。

### 3.2 两项辅助损失

**分组级路由监督。** 小核的监督信号是「该用哪一组专家」而非「该用第几个专家」——组内做 logsumexp 池化后再算交叉熵：

```python
group_logits = torch.stack([
    torch.logsumexp(little_logits[..., lo:hi], -1) - math.log(hi - lo)
    for lo, hi in GROUP_BOUNDS
], -1)
```

减去 `log(size)` 是必需的：各组大小不等（8 / 8 / 16），裸 logsumexp 会系统性偏向大组。

**负载均衡损失。** Switch Transformer 风格，防止路由塌缩到少数专家：

```python
frac = topk_mask.mean(dim=0)      # 专家实际承接比例
mean_p = probs.mean(dim=0)        # 路由器平均概率
loss = num_experts * (frac * mean_p).sum()
```

### 3.3 Tier-3 流式搬运

专家常驻主机 pinned 内存，每个 token 从每层选 Top-8 搬入显存 compute buffer。三个关键点：

**常驻 staging buffer。** 不用 `caching allocator` 分配临时张量。这消除了一类真实的数据竞争：早期版本在 `transfer_stream` 上分配张量、在默认流上使用、随即出作用域被回收，下一轮 `.to()` 可能覆写正在被 matmul 读的数据。实测早期版本 40 次相同前向抖动 1.56e-2，修复后为 0.00。

**双缓冲 ping-pong。** 两组 staging buffer 交替使用，每组配一个 `torch.cuda.Event`：

```python
self.transfer_stream.wait_event(self.staging_events[slot])   # 等两拍前的计算读完
with torch.cuda.stream(self.transfer_stream):
    for j, eid in enumerate(selected_ids):
        self.staging_A[slot][j].copy_(self.host_lora_A[eid], non_blocking=True)
        self.staging_B[slot][j].copy_(self.host_lora_B[eid], non_blocking=True)
torch.cuda.current_stream().wait_stream(self.transfer_stream)
```

**遥测不打断流水线。** 路由统计全部在 GPU 上累加，每层每 token 只做一次 D2H——而那次同步本来就是要把专家下标取回主机内存发起搬运的，顺路把权重一起带回来。相比每层两次 `.item()`，吞吐提升 15.4%。

### 3.4 训练配置

| 参数组 | 学习率 | 说明 |
| :--- | :--- | :--- |
| Tier-2 理科大核 | 2e-5 | 微火慢炖 |
| 路由器 | 3e-4 | 快速对齐 |
| Tier-3 微专家 | 5e-4 | LoRA 快速学习 |

```python
total_loss = lm_loss                                    # 只监督 response，prompt 掩码
           + 0.1 * (big_router_loss / num_layers)       # 大核域分类
           + 0.1 * (little_router_loss / num_layers)    # 小核分组分类
           + 0.01 * (load_balance_loss / num_layers)    # 专家负载均衡
```

---

## 4. 已知局限

如实记录，避免误判：

- **`<think>` 推理链丢失。** 训练语料 8000 条中 0 条含 `<think>`，微调后模型直接给答案。这是数据选择的结果而非代码缺陷，但确实牺牲了 Qwen3 原生的推理链能力。困惑度换来了这个取舍。
- **组间分布不均。** 10 prompt 累计下，代码组拿到 24.8% 调用、数学组 12.7%、写作组 62.4%（写作组有 16 个专家，天然占 50% 容量）。组内无饿死，但组间偏斜，`LOAD_BALANCE_WEIGHT` 可能需要调大。
- **第 0 层路由较弱。** 判别力 76.8% vs 41.1%，明显弱于后续各层。逐层看分化是清晰的，但跨层平均会被第 0 层拉低。
- **Tier-3 搬运有实打实的开销。** 去掉小核后 decode 从 38.3ms 降到 27.7ms，Tier-3 占 28%。这是 host-RAM 流式架构的固有代价——相比 GPU 驻留的同规模 MoE，换来的是 56 MiB 主机内存而非数 GB 显存。
- **prefill 路由不对称。** 训练时小核逐位置选专家，推理时一个 prompt 只按末 token 选一组专家（否则 prompt 有多少 token 就要搬多少轮专家）。decode 阶段两者一致。

---

## 5. 快速开始

### 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch, transformers, accelerate, datasets
```

需要一块 CUDA 显卡。开发与实测环境：RTX 5090 D + PyTorch 2.14 + transformers 5.17。

### 跑起来

```bash
# 1. 准备语料（仓库已附带 dual_contrast_data.jsonl，可跳过）
python prepare_dual_data.py

# 2. 训练（约 5 分钟）
python train_dual_big_resurrect.py

# 3. 交互对话
python chat_dual_big_resurrect.py
```

对话终端支持：

- `clear` — 重置对话记忆
- `exit` / `quit` — 退出
- `Ctrl+C` — 中断当前生成

每次回答后会打印双核能量分配与微专家调用分布。

---

## 6. 文件说明

| 文件 | 作用 |
| :--- | :--- |
| `prepare_dual_data.py` | 从 HF 拉取并拼接 4000 理科 + 4000 文科对抗语料 |
| `train_dual_big_resurrect.py` | 植入双大核架构，分级学习率训练 |
| `chat_dual_big_resurrect.py` | 加载权重、PCIe 流式推理、交互终端 |
| `dual_contrast_data.jsonl` | 8000 条训练语料（已附带） |

`dual_big_resurrect_weights.pt` 由训练生成，不入版本库。

---

## 7. 引用

```bibtex
@misc{dualbiglittle_moe_2026,
  author = {aifeifei798 and Community Contributors},
  title = {DualBigLittle-MoE: A Tri-Tier Hierarchical MoE Architecture with Dual Dense VRAM Cores and Streaming Micro-Expert Clusters},
  year = {2026},
  publisher = {GitHub and Hugging Face},
  howpublished = {\url{https://github.com/aifeifei798/DualBigLittle-MoE}}
}
```

## 许可

[Apache 2.0](LICENSE)
