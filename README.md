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

DualBigLittle-MoE 借鉴移动端三簇 CPU（大核 / 小核 / 能效核），把模型容量拆到三个物理位置：

| 层 | 位置 | 内容 | 作用 |
| :--- | :--- | :--- | :--- |
| **Tier 1** 文科锚核 | GPU 显存 | 原版 MLP，**完全冻结** | 语言底座，保证不退化 |
| **Tier 2** 理科孪生核 | GPU 显存 | 克隆 MLP，低学习率微调 | 代码、数学、逻辑 |
| **Tier 3** 微专家池 | 主机 pinned RAM | 896 个 rank-16 LoRA，**54 MiB** | 细粒度专家，按需流式搬运 |

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

全部数字由 `eval_ppl.py` 生成，可在本地复现（见 §5）。开发与实测环境：
**RTX 5090 D · PyTorch 2.14.0+cu130 · transformers 5.17.0 · Python 3.12**。

### 2.1 分域困惑度（验证集，754 条，未参与训练）

`python eval_ppl.py --weights checkpoints/v3_full.pt --split val`

| 领域 | 原生 Qwen3-0.6B | 双大核 | 变化 |
| :--- | ---: | ---: | ---: |
| Code | 7.58 | **4.54** | −40.1% |
| Math | 6.27 | **3.73** | −40.5% |
| Arts | 8.90 | **5.49** | −38.2% |

三个领域均匀改善约 40%。语料按 domain 分层切分 90/10，验证集与训练集无重叠。

> 旧版 README 声称的「Code −55% / Math −51% / Arts −40%」是在**未切分的训练语料**上测得的
> in-distribution 指标，天然偏乐观。本表是首次在真实留出集上给出的数字。

### 2.2 专家健康度

| 指标 | 数值 |
| :--- | :--- |
| 获得非零权重的 (层, 专家) 对 | **896 / 896** |
| 完全未被调用的 | **0 / 896** |
| 存在饿死专家的层数 | **0 / 28** |
| 单对最小权重占比 | 3.2e-6 % |

### 2.3 数值稳定性

| 指标 | 数值 |
| :--- | :--- |
| 20 次相同前向的输出差异 | **0.00**（逐位一致） |
| 训练路径 vs 推理路径（decode） | **数值等价**（单元测试保证） |

### 2.4 逐层路由分化 · 理科大核权重均值

```
   层      Code    Math    Arts
   0      0.700   0.614   0.363
   8      0.476   0.473   0.504
  16      0.480   0.412   0.246
  27      0.466   0.315   0.591
```

**如实说明：路由分化比旧版 README 描述的弱得多。** 旧版声称末层
「Code 95.9% vs Arts 0.7%」，那是在未切分语料上测得的；换用验证集与
清洗后语料，理科核权重普遍落在 0.3~0.7，领域区分度明显下降。
第 0 层的分化反而最明显（Code 0.70 vs Arts 0.36）。
**大核的收益主要来自「多了一个可微调副本」，而不是「学到了清晰的领域分工」。**

### 2.5 训练

| 指标 | 数值 |
| :--- | :--- |
| 语料 | 6795 条（去重 + 长度过滤后） |
| 优化步数 | 424 |
| 耗时 | 4.4 分钟 |
| 吞吐 | 24~26 samples/s |

### 2.6 checkpoint 体积

| 格式 | 体积 | 量化相对误差 | 验证集 PPL (Code/Math/Arts) |
| :--- | ---: | ---: | :--- |
| Tier-2 全量副本 | 561.9 MiB | — | 5.060 / 3.305 / 4.380 |
| **int8 量化差分** | **310.7 MiB** | 6.3e-5 | 5.059 / 3.304 / 4.379 |

int8 差分让体积降 45%，困惑度变化在第 4 位小数，实践中不可分辨。

> **注意**：bf16 存差分**不会**减小体积。实测 Tier-2 相对基座的 Δ 只占权重的
> 0.11%~0.21%，但差分并不稀疏，28 层仍是 504 MiB。真正要缩体积必须量化。
> 详见 §4.4。

---

## 3. 关键实现

### 3.1 单一事实源

超参此前散落在训练与推理脚本两处，靠注释「必须与…保持一致」人工同步 ——
改一个忘另一个就是静默的数值错位。现在：

- `dbl/config.py` 是全部超参的唯一定义
- `dbl/groups.py` 是专家分组拓扑的唯一定义
- **checkpoint 自带 `config`**，推理端从文件反序列化重建，
  不再需要维护任何第二份常量

### 3.2 微专家用堆叠权重

32 个 LoRA 存成两块 `(E, r, D)` 张量而非 32 个 `nn.Module`：

```python
h = torch.matmul(x.reshape(n, dim), a_flat.t())   # (N, E*r)
h = h * topk_mask_expanded                        # top-k 归一化权重
y = torch.matmul(h, b_flat)                       # (N, D)
```

不物化 `(N, E, D)` 激活，显存可控。

### 3.3 防止专家饿死的真正机制

早期版本用 `ModuleList` + 单专家索引，只有被选中的专家有梯度，其余因
`lora_B` 零初始化且无梯度而**输出恒为 0**——名义 896 个专家，实际只有
少数能工作。

改为堆叠权重解决了 ModuleList 的路径切断问题，但**「稠密前向让所有专家
拿到梯度」这个说法是不准确的**（见 `tests/test_moe.py`）：

| | LM loss 单独 | 负载均衡 + 分组监督 |
| :--- | :--- | :--- |
| `lora_B` 拿到梯度的专家 | 仅被 top-k 选中的 | — |
| `router_little` 梯度 | **0**（`B` 零初始化时） | **全部行非零** |

top-k 掩码会把未选中专家的贡献归零，其梯度自然也是零。真正防止路由
塌缩的是**负载均衡损失与分组级监督**，让路由器的每一行都保持可训练。
这也是 `router_aux_weight` 不能设为 0 的原因。

### 3.4 两项辅助损失

**分组级路由监督。** 小核的监督信号是「该用哪一组专家」而非「该用第几个
专家」——组内做 `logsumexp` 池化后再算交叉熵：

```python
group_logits = torch.stack([
    torch.logsumexp(little_logits[..., lo:hi], -1) - math.log(hi - lo)
    for lo, hi in groups.bounds
], -1)
```

减去 `log(size)` 是必需的：各组大小不等（8 / 8 / 16），裸 `logsumexp`
会系统性偏向大组。

**负载均衡损失。** Switch Transformer 风格：

```python
frac   = topk_mask.mean(dim=0)      # 专家实际承接比例
mean_p = probs.mean(dim=0)          # 路由器平均概率
loss   = num_experts * (frac * mean_p).sum()
```

### 3.5 Tier-3 流式搬运

专家常驻主机 pinned 内存，prefill 时按**末 token** 选一次专家集合，
逐 token 搬运到 compute buffer。三个关键点：

**常驻 staging buffer。** 不走 caching allocator 分配临时张量。这消除了一类
真实竞态：在 `transfer_stream` 上分配、在默认流上使用、随即被回收，下一轮
`.to()` 可能覆写正在被 matmul 读的数据。

**双缓冲 ping-pong。** 两组 staging buffer 交替使用，每组配一个 `torch.cuda.Event`：

```python
self.transfer_stream.wait_event(self.staging_events[slot])   # 等两拍前的计算读完
with torch.cuda.stream(self.transfer_stream):
    for j, eid in enumerate(selected):
        self.staging_A[slot][j].copy_(self.host_lora_A[eid], non_blocking=True)
        self.staging_B[slot][j].copy_(self.host_lora_B[eid], non_blocking=True)
torch.cuda.current_stream().wait_stream(self.transfer_stream)
```

**遥测不打断流水线。** 路由统计全部在 GPU 上累加，每层每 token 只做一次
D2H——而那次同步本来就是要把专家下标取回主机内存发起搬运的。

### 3.6 训练配置

| 参数组 | 学习率 | 说明 |
| :--- | :--- | :--- |
| Tier-2 理科大核 | 2e-5 | 微火慢炖 |
| 路由器 | 3e-4 | 快速对齐 |
| Tier-3 微专家 | 5e-4 | LoRA 快速学习 |

```python
total_loss = lm_loss                                    # 只监督 response，prompt 掩码
           + 0.1 * big_router_loss                     # 大核域分类
           + 0.1 * little_router_loss                  # 小核分组分类
           + 0.01 * load_balance_loss                  # 专家负载均衡
```

---

## 4. 已知局限

如实记录，避免误判：

- **prefill 路由不对称。** 训练时小核逐位置选专家，推理时一个 prompt 只按
  末 token 选一组专家（否则 prompt 有多少 token 就要搬多少轮专家）。
  实测这会造成显著差异：验证集上训练路径 PPL 4.54/3.73/5.49，推理路径
  6.57/5.51/7.93。**decode 阶段两者等价**（由 `tests/test_moe.py` 保证），
  但长 prompt 的首 token 输出会受影响。这是当前最值得改进的一处。
- **领域分工不清晰。** 见 §2.4。路由权重没有形成 README 早期版本所描述的
  清晰分野，大核的收益更多来自容量增加而非领域区分。
- **组间分布不均。** 写作组有 16 个专家，天然占 50% 容量，调用占比偏高。
  组内无饿死，但组间偏斜，`load_balance_weight` 可能需要调大。
- **`<think>` 推理链丢失。** 训练语料中 0 条含 `<think>`，微调后模型直接给
  答案。这是数据选择的结果而非代码缺陷，但确实牺牲了 Qwen3 原生的推理链能力。
- **验证集偏小。** 754 条，指标方差不可忽略；§2.1 的 150/domain 采样在
  ±0.1 PPL 量级上仍有噪声。
- **int8 差分依赖基座逐位一致。** checkpoint 会校验基座权重指纹，不匹配时
  直接报错而非静默还原出错误权重。换基座需重新训练或重存。

---

## 5. 快速开始

### 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 锁定版本
pip install -r requirements-dev.txt      # 测试与 lint
```

需要一块 CUDA 显卡。

### 跑起来

```bash
# 1. 准备语料（仓库已附带，可跳过）
python prepare_dual_data.py

# 2. 训练（约 4.5 分钟）
python train_dual_big_resurrect.py --out checkpoints/v3_full.pt

# 3. 交互对话
python chat_dual_big_resurrect.py --weights checkpoints/v3_full.pt

# 4. 复现 README 的全部数字
python eval_ppl.py --weights checkpoints/v3_full.pt --split val
```

对话终端支持 `clear`（重置记忆）、`exit` / `quit`（退出）、`Ctrl+C`（中断生成）。
每次回答后会打印双核能量分配与微专家调用分布。

### 存储优化

```bash
# 体积降 45%，PPL 变化在第 4 位小数
python train_dual_big_resurrect.py --out checkpoints/v3_int8.pt --delta-dtype int8
```

### 迁移旧 checkpoint

```bash
python -m dbl.migrate_ckpt --src old_weights.pt --dst new.pt --delta-dtype int8
```

### 测试

```bash
pytest                    # 83 项，约 1.5 秒，无需 GPU
ruff check .
```

---

## 6. 文件说明

| 文件 | 作用 |
| :--- | :--- |
| `dbl/config.py` | **全部超参的单一事实源** |
| `dbl/groups.py` | **专家分组拓扑的唯一定义** |
| `dbl/moe.py` | `DualBigLittleMoE` 基类 + `TrainMoE` / `InferMoE` |
| `dbl/data.py` | 语料加载与标签掩码 |
| `dbl/checkpoint.py` | 带 config 元信息的保存/加载、int8 量化差分 |
| `dbl/train.py` | 训练循环（种子、logging、resume） |
| `dbl/prepare.py` | 语料装配（清洗、去重、切分） |
| `dbl/migrate_ckpt.py` | 旧格式 checkpoint 迁移 |
| `prepare_dual_data.py` | 语料装配入口 |
| `train_dual_big_resurrect.py` | 训练入口 |
| `chat_dual_big_resurrect.py` | 交互对话入口 |
| `eval_ppl.py` | **分域评估与可复现性体检** |
| `tests/` | 83 项测试 |

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
