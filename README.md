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

> **数据口径声明。** 本节所有表格均来自
> `python eval_ppl.py --split val --samples-per-domain 150`（域内**分层等距**采样）。
> 早期版本的对照实验使用「取前 n 条」采样，与本节数字**不可比**——同一权重下
> Arts PPL 分别为 5.49 与 12.46。引用时请勿混用。

### 2.1 分域困惑度（验证集，754 条，未参与训练）

`python eval_ppl.py --weights checkpoints/v5_auxfix.pt --split val`

| 领域 | 原生 Qwen3-0.6B | 双大核 | 变化 |
| :--- | ---: | ---: | ---: |
| Code | 7.58 | **6.80** | −10.4% |
| Math | 6.27 | **5.63** | −10.3% |
| Arts | 8.90 | **8.20** | −7.8% |

平均 **−9.5%**。这是修复辅助损失 bug 后的诚实数字 —— 路由真正可用时的
PPL 表现，代价与收益的完整分析见 §4.1。

> 旧版 README 声称的「Code −55% / Math −51% / Arts −40%」是在**未切分的训练
> 语料**上测得的 in-distribution 指标，天然偏乐观。

### 2.2 专家健康度

| 指标 | 数值 |
| :--- | :--- |
| 获得非零权重的 (层, 专家) 对 | **896 / 896** |
| 完全未被调用的 | **0 / 896** |
| 存在饿死专家的层数 | **0 / 28** |
| 单对最小权重占比 | 1.3e-4 % |

### 2.3 数值稳定性

| 指标 | 数值 |
| :--- | :--- |
| 20 次相同前向的输出差异 | **0.00**（逐位一致） |
| 训练路径 vs 推理路径（decode） | **数值等价**（单元测试保证） |

### 2.4 逐层路由分化 · 理科大核权重均值

修复辅助损失稀释 bug 前后对比（同为验证集，域内分层采样）：

```
          修复前(bug)              修复后
   层     Code  Math  Arts      Code  Math  Arts
   0     0.700 0.614 0.363     0.812 0.703 0.219
   8     0.476 0.473 0.504     0.621 0.724 0.261
  16     0.480 0.412 0.246     0.626 0.655 0.248
  27     0.466 0.315 0.591     0.885 0.948 0.202
          ^ 分化倒挂               ^ 分化清晰
```

末层（layer 27）大核路由 top-1 命中率：

| | Code | Math | Arts |
| :--- | ---: | ---: | ---: |
| 修复前 | 46.4% | 34.1% | 34.4% |
| **修复后** | **91.6%** | **97.7%** | **82.3%** |

修复前三个域的命中率都在 34~46%，即**等同随机猜测**——根因是辅助损失被
`/num_layers` 稀释了 28 倍（详见 §3.3）。修复后路由才真正工作。

> 但路由变好带来一个代价：文科 PPL 从 5.49 退到 8.20。完整分析见 §4.1。

### 2.5 训练

| 指标 | 数值 |
| :--- | :--- |
| 语料 | 6795 条（去重 + 长度过滤后） |
| 优化步数 | 424 |
| 耗时 | 4.4 分钟 |
| 吞吐 | 24~26 samples/s |

### 2.6 checkpoint 体积

同一权重（`v2_full`）的两种存储格式：

| 格式 | 体积 | 量化相对误差 | PPL (Code/Math/Arts) |
| :--- | ---: | ---: | :--- |
| Tier-2 全量副本 | 561.9 MiB | — | 5.060 / 3.305 / 4.380 |
| **int8 量化差分** | **310.7 MiB** | 6.3e-5 | 5.059 / 3.304 / 4.379 |

int8 差分让体积降 45%，困惑度变化在第 4 位小数，实践中不可分辨。

> 本表测的是**存储格式**的差异，用 `v2_full` 权重在**训练语料切片**上评估
> （`--split all`，n=150/domain），因此 PPL 绝对值与 §2.1 的验证集数字
> 不可比 —— 这里只看两种格式之间的相对差异。

> **注意**：bf16 存差分**不会**减小体积。实测 Tier-2 相对基座的 Δ 只占权重的
> 0.11%~0.21%，但差分并不稀疏，28 层仍是 504 MiB。真正要缩体积必须量化。

---

## 3. 关键实现

### 3.1 单一事实源

超参此前散落在训练与推理脚本两处，靠注释「必须与…保持一致」人工同步 ——
改一个忘另一个就是静默的数值错位。现在：

- `dbl/config.py` 是全部超参的唯一定义
- `dbl/groups.py` 是专家分组拓扑的唯一定义
- `dbl/runtime.py` 是设备解析的唯一定义
- **checkpoint 自带 `config`**，推理端从文件反序列化重建，
  不再需要维护任何第二份常量

### 3.1b 设备解析只有一个入口

设备串此前硬编码在各脚本里（`DEV = "cuda:0"`），带来三类静默故障：

1. 无显卡机器上跑到一半才炸，报错指向 `torch.cuda.Stream` 而非「没装 CUDA」
2. `cuda:1` 不存在时，错误来自 `from_pretrained` 深处的调用栈
3. 显卡算力不在 torch 预编译列表内，退化成 PTX JIT，
   表现为「卡住了」而不是「不兼容」

现在 `dbl/runtime.py` 把这些收敛到 `resolve_device()`，在**建模之前**完成校验：

- `auto` 选当前最空的一张卡，多卡机上不固定 `cuda:0`
- 训练 / 评估入口传 `allow_cpu=False`，**禁止静默退回 CPU** ——
  否则「本该上 GPU 却拿到 CPU 的慢结果」看起来完全正常
- 卡号写错（`cuda:9`）直接报「只检测到 N 张 GPU」，不降级、不加载模型
- 算力不在 arch 列表时提前警告

回归测试见 `tests/test_runtime.py`：用 monkeypatch 模拟「无卡 / 少卡 /
算力不匹配」，因此这些分支在 CI 的 CPU 机器上也能覆盖。

### 3.2 微专家用堆叠权重

32 个 LoRA 存成两块 `(E, r, D)` 张量而非 32 个 `nn.Module`：

```python
h = torch.matmul(x.reshape(n, dim), a_flat.t())   # (N, E*r)
h = h * topk_mask_expanded                        # top-k 归一化权重
y = torch.matmul(h, b_flat)                       # (N, D)
```

不物化 `(N, E, D)` 激活，显存可控。

### 3.3 辅助损失不做层间平均（重要）

历史实现把 28 层的辅助 loss 除以 `num_layers`：

```python
big_loss = big_loss / len(modules)      # ❌ 稀释 28 倍
```

但每层的 router 都是**独立参数**。对第 i 层的 `router_big` 求导时，
系数是 `0.1 / 28 = 0.0036` —— 辅助信号被稀释 28 倍，等效于把
`router_aux_weight` 从 0.1 悄悄降到 0.0036。

症状极其隐蔽：训练不报错、总 loss 正常下降、专家健康度 896/896 全绿，
但只有辅助损失长期停在随机猜测水平之上。实测 60 步后大核路由准确率
`50.0% → 50.1%`，等同随机。

修复后（末层大核路由 top-1 命中率）：

| | 修复前 | 修复后 |
| :--- | :---: | :---: |
| Code | 46.4% | **91.6%** |
| Math | 34.1% | **97.7%** |
| Arts | 34.4% | **82.3%** |
| 末层 Code−Arts 权重分化 | −0.125 | **+0.683** |

回归测试见 `tests/test_aux_loss.py`（含反证：恢复旧实现会失败）。

> 修复让路由真正工作，但也暴露出 §4.1 的架构冲突：路由学准了，
> 文科 PPL 反而变差（5.49 → 8.20）。这是真实取舍，不是可以两全的调参问题。

### 3.4 防止专家饿死的真正机制

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

### 3.5 两项辅助损失

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

### 3.6 Tier-3 流式搬运

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

### 3.7 训练配置

| 参数组 | 学习率 | 说明 |
| :--- | :--- | :--- |
| Tier-2 理科大核 | 2e-5 | 微火慢炖（实测提到 1e-4 几乎无收益，见 §4.1） |
| 路由器 | 3e-4 | 快速对齐 |
| Tier-3 微专家 | 5e-4 | LoRA 快速学习 |

```python
total_loss = lm_loss                                    # 只监督 response，prompt 掩码
           + 0.1 * (Σ_layer big_router_loss)           # 大核域分类（不除层数）
           + 0.1 * (Σ_layer little_router_loss)        # 小核分组分类（不除层数）
           + 0.01 * (Σ_layer load_balance_loss)        # 专家负载均衡
```

`router_aux_weight` 可用 `--router-aux-weight` 覆盖。调高它会让路由更准，
但按 §4.1 会牺牲文科 PPL——这是一个需要显式选择的取舍，不是「越大越好」。

---

## 4. 已知局限

### 4.1 架构设计冲突：路由锐度 vs 文科能力

**这是本项目最重要的发现，也是当前最大的未解问题。**

架构假设「每个领域有专属的正确路由」，于是辅助损失把 Arts 推向
`w_sci ≈ 0`（走文科核）。但 **Tier-1 是冻结的**——文科除了「不用理科核」
之外没有任何可学习的专属容量。路由一旦学对，Arts 就退化为原版模型，
微调收益归零。

实测（同口径：`--split val --samples-per-domain 150`，域内分层采样）：

| 训练配置 | 末层路由命中率 (C/M/A) | Code PPL | Math PPL | Arts PPL | 平均变化 |
| :--- | :---: | ---: | ---: | ---: | ---: |
| `aux_w=0.0036`（修复前，路由等同随机） | 46/34/34% | 4.54 | 3.73 | **5.49** | **−39.6%** |
| `aux_w=0.1`（修复后，路由可用） | 92/98/82% | 6.80 | 5.63 | **8.20** | **−9.5%** |

两点结论：

1. **PPL 与路由锐度负相关。** 路由从随机变成可用（命中率 34% → 82%），
   代价是三域 PPL 全面变差，尤其文科（5.49 → 8.20）。
2. **决定性变量是 Arts 的 `w_sci`。** 弱路由时 Arts 仍有一半权重落在
   Tier-2 上（0.45~0.53），强路由时被压到 0.20。Tier-1 冻结，
   放弃 Tier-2 就等于放弃学习。

换句话说：**在这个架构里，PPL 收益来自「Tier-2 在所有领域都参与」，
而不是来自「领域分工」。** 让路由学得更好，反而削弱了唯一的学习来源。

补充实验表明**这不是容量不足**（另一组对照，取前 n 条采样，仅供组内比较）：
把 `lr_big_sci` 从 2e-5 提到 1e-4，Arts PPL 仅 20.37 → 19.61，
几乎无改善。瓶颈不在学习率，在 `w_sci` 被压低这件事本身。

可能的解法（均未实现）：

- 给文科也加一个**可学习**的专属容量（解冻 Tier-1 或增加 Tier-4），
  消除「路由到文科核 = 放弃学习」这一根本矛盾
- 把 `big_target` 监督改成软标签 / 只约束理科侧，不强制 Arts 远离 Tier-2
- 把 `w_sci` 设为带下限的区间，如 `w_sci ∈ [0.3, 0.7]`

### 4.2 其他局限

- **prefill 路由不对称。** 训练时小核逐位置选专家，推理时一个 prompt 只按
  末 token 选一组专家（否则 prompt 有多少 token 就要搬多少轮专家）。
  实测会造成显著差异：验证集上训练路径 PPL 6.80/5.63/8.20，推理路径
  7.95/7.29/10.85。**decode 阶段两者等价**（由 `tests/test_moe.py` 保证），
  但长 prompt 的首 token 输出会受影响。
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

推荐用 uv（会自动读取 `pyproject.toml` 里的 cu130 索引并锁定版本）：

```bash
uv sync                # 按 uv.lock 装齐，含 torch 2.14.0+cu130
uv sync --group dev    # 额外装 pytest / ruff
```

或用 pip：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url \
    https://download.pytorch.org/whl/cu130
pip install -r requirements-dev.txt
```

**torch 必须是 CUDA 版。** 装成 CPU-only 版时，训练要等到
`torch.cuda.Stream` 才报错，栈底信息完全指不到「装错包」这个根因。
`pyproject.toml` 显式指定了 cu130 源并锁版本，就是为了避免这一步出错。

### 环境体检

```bash
python doctor.py              # 依赖 / CUDA / 设备参数解析
python doctor.py --gpu-check  # 额外加载模型，跑一次真实前向
```

`doctor.py` 会确认 torch 是不是 CUDA 版、当前卡算力是否在预编译 arch 列表内
（不在则退回 PTX JIT，首个 kernel 会明显变慢），以及设备字符串能否正确解析。

### 设备选择

所有入口的 `--device` 都接受 `auto`（默认）/ `cpu` / `cuda` / `cuda:N`，
由 `dbl/runtime.py` 统一解析：

- `auto` **选当前最空的一张卡**（多卡机上不固定 `cuda:0`）
- 训练与评估入口**不会静默退回 CPU**——显式要求 GPU 而环境不满足时
  直接报错并给出修复命令
- `cpu` 保留给单元测试与无显卡调试
- 也可用环境变量 `DBL_DEVICE=cuda:1` 一次性覆盖

设备串在建模之前就完成校验，因此 `--device cuda:9` 这类笔误会立刻
以 `请求 'cuda:9'，但只检测到 1 张 GPU` 结束，而不是加载完 0.6B 模型
才崩。

### 跑起来

```bash
# 1. 准备语料（仓库已附带，可跳过）
python prepare_dual_data.py

# 2. 训练（约 4.5 分钟）
python train_dual_big_resurrect.py --out checkpoints/demo.pt

# 3. 交互对话
python chat_dual_big_resurrect.py --weights checkpoints/demo.pt

# 4. 复现 README §2 的全部数字
python eval_ppl.py --weights checkpoints/demo.pt --split val
```

以上命令使用默认超参（`router_aux_weight=0.1`），对应 §2.1 的
**−10.4% / −10.3% / −7.8%**。若想复现 §4.1 中 PPL 更好但路由等同随机的
那一行（旧行为），加 `--router-aux-weight 0.0036`。

对话终端支持 `clear`（重置记忆）、`exit` / `quit`（退出）、`Ctrl+C`（中断生成）。
每次回答后会打印双核能量分配与微专家调用分布。

### 存储优化

```bash
# 体积降 45%，PPL 变化在第 4 位小数
python train_dual_big_resurrect.py --out checkpoints/demo_int8.pt --delta-dtype int8
```

### 迁移旧 checkpoint

```bash
python -m dbl.migrate_ckpt --src old_weights.pt --dst new.pt --delta-dtype int8
```

### 导出为 Hugging Face 仓库

```bash
python export_to_hf.py            # 基座 + checkpoint -> ./hf_export
python export_to_hf.py --pool-location device   # 跳过 PCIe 搬运
```

产物自带 `auto_map`，外部用户标准两行加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("./hf_export", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./hf_export", trust_remote_code=True)
```

`modeling_dualbig_moe.py` 只依赖 `torch` 与 `transformers`，**不 import 本仓库的
`dbl`**——远程代码只会被拷进动态模块缓存，内部 import 必然断掉。注意力 / RoPE /
KV cache 直接复用 transformers 的 Qwen3 实现（升级时只需核对三个 import），
本仓库只替换每层 FFN 并实现 Tier-3 搬运。

导出与原版推理管线的 logits **逐位相同**（`max|diff| = 0`，含 greedy 生成），
回归测试见 `tests/test_hf_export.py`。

### 测试

```bash
pytest                    # 142 项，约 3 秒（HF 导出的 8 项需 GPU）
ruff check .
```

---

## 6. 文件说明

| 文件 | 作用 |
| :--- | :--- |
| `dbl/config.py` | **全部超参的单一事实源** |
| `dbl/groups.py` | **专家分组拓扑的唯一定义** |
| `dbl/runtime.py` | **CUDA 设备解析与环境体检**（auto / cuda:N、arch 校验） |
| `dbl/moe.py` | `DualBigLittleMoE` 基类 + `TrainMoE` / `InferMoE` |
| `dbl/data.py` | 语料加载与标签掩码 |
| `dbl/checkpoint.py` | 带 config 元信息的保存/加载、int8 量化差分 |
| `dbl/train.py` | 训练循环（种子、logging、resume） |
| `dbl/prepare.py` | 语料装配（清洗、去重、切分） |
| `dbl/migrate_ckpt.py` | 旧格式 checkpoint 迁移 |
| `configuration_dualbig_moe.py` | **HF 导出的配置类**（自包含全部骨架字段） |
| `modeling_dualbig_moe.py` | **HF 导出的模型实现**（自包含，零 `dbl` 依赖） |
| `export_to_hf.py` | **HF 仓库导出入口**（权重合并 + auto_map + 卡片） |
| `example_usage.py` | 导出目录附带的上手示例 |
| `prepare_dual_data.py` | 语料装配入口 |
| `doctor.py` | **环境体检入口**（一条命令确认跑在 CUDA 上） |
| `train_dual_big_resurrect.py` | 训练入口 |
| `chat_dual_big_resurrect.py` | 交互对话入口 |
| `eval_ppl.py` | **分域评估与可复现性体检** |
| `tests/` | 142 项测试 |

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
