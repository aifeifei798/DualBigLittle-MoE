import copy
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# 0. 全局超参
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-0.6B"
MICRO_BATCH = 4
GRAD_ACCUM_STEPS = 4
MAX_LENGTH = 512
TOP_K = 8
GAMMA = 0.3
LORA_RANK = 16
LORA_ALPHA = 16.0
NUM_EXPERTS = 32
ROUTER_AUX_WEIGHT = 0.1
LOAD_BALANCE_WEIGHT = 0.01

# 专家分组（大小不等，显式给边界）：[0:8] 代码, [8:16] 数学, [16:32] 写作
DOMAIN_GROUP = {"Code": 0, "Math": 1, "Arts": 2}
GROUP_BOUNDS = [(0, 8), (8, 16), (16, 32)]
NUM_GROUPS = len(GROUP_BOUNDS)
assert GROUP_BOUNDS[-1][1] == NUM_EXPERTS, \
    f"分组边界必须覆盖 0..{NUM_EXPERTS}"


# ----------------------------------------------------------------------
# 1. 数据集加载
# ----------------------------------------------------------------------
class DualContrastDataset(Dataset):
    """文理 1:1 对抗语料。

    标签约定：
      big_target  : 0 -> 文科大核, 1 -> 理科大核
      group_target: 0 -> 代码专家组(0-7), 1 -> 数学(8-15), 2 -> 写作(16-31)

    注意：``labels`` 只监督 response 段，prompt 段全部置 -100。
    """

    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        prompt_text = (f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        prompt_ids = self.tokenizer(prompt_text,
                                    add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt_text + item["response"] +
                                  "<|im_end|>\n",
                                  add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:self.max_length]

        # 只监督 response：prompt 部分不计 loss
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_prompt + full_ids[n_prompt:]

        pad_id = self.tokenizer.pad_token_id
        pad_len = self.max_length - len(full_ids)
        input_ids = full_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len

        group = item.get("little_group")
        if group is None:
            # 兼容旧语料：little_target 存的是分组起始下标 (0/8/16)
            group = DOMAIN_GROUP.get(item.get("domain"))
            if group is None:
                lt = int(item.get("little_target", 0))
                group = next((g for g, (lo, hi) in enumerate(GROUP_BOUNDS)
                              if lo <= lt < hi), 0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask":
            torch.tensor(attention_mask, dtype=torch.long),
            "labels":
            torch.tensor(labels, dtype=torch.long),
            "big_target":
            torch.tensor(item["big_target"], dtype=torch.long),
            "group_target":
            torch.tensor(int(group), dtype=torch.long),
        }


# ----------------------------------------------------------------------
# 2. 架构定义（双大核安全混音 + 32 路堆叠微专家）
# ----------------------------------------------------------------------
class DualBigResurrectWrapper(nn.Module):
    """把原生 MLP 包装成「大核双权重 + 小核专家池」。

    微专家不再是一堆 nn.Module，而是两块堆叠权重：
        lora_A : (E, r, D)   h = x @ A.T
        lora_B : (E, r, D)   y = h @ B
    这样前向可以塌缩成两次大 matmul，而状态字典依然紧凑。
    """

    def __init__(self,
                 original_mlp,
                 hidden_dim,
                 rank=LORA_RANK,
                 lora_alpha=LORA_ALPHA,
                 num_experts=NUM_EXPERTS,
                 top_k=TOP_K,
                 gamma=GAMMA,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gamma = gamma
        self.scaling = lora_alpha / rank

        # 文科大核：继承原版，死死锁死
        self.big_arts = original_mlp
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # 理科大核：克隆出来，接受温和微调
        self.big_sci = copy.deepcopy(original_mlp).to(device)
        for p in self.big_sci.parameters():
            p.requires_grad = True

        # 文理大核路由器
        self.router_big = nn.Linear(hidden_dim,
                                    2,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        self.router_little = nn.Linear(hidden_dim,
                                       num_experts,
                                       bias=False,
                                       device=device,
                                       dtype=dtype)

        # 32 路微专家池（堆叠参数）
        self.lora_A = nn.Parameter(
            torch.empty(num_experts, rank, hidden_dim, device=device,
                        dtype=dtype))
        self.lora_B = nn.Parameter(
            torch.empty(num_experts, rank, hidden_dim, device=device,
                        dtype=dtype))
        # A 用 kaiming（fan_in 视作 hidden_dim），B 置零 -> 初始输出恒为 0
        nn.init.kaiming_uniform_(self.lora_A.view(-1, hidden_dim),
                                 a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 供外层算辅助 loss 的缓存
        self.last_router_big_logits = None
        self.last_router_little_logits = None
        self.last_little_probs = None
        self.last_little_topk_w = None

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen

        # --- 大核双引擎（逐位置路由，训练/推理完全一致）---
        router_big_logits = self.router_big(x)
        self.last_router_big_logits = router_big_logits
        weights_big = torch.softmax(router_big_logits.float(),
                                    dim=-1).to(x.dtype)

        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] *
                                                        sci_out)

        # --- 小核专家池 ---
        router_little_logits = self.router_little(x)
        self.last_router_little_logits = router_little_logits

        probs = torch.softmax(router_little_logits.float(), dim=-1).to(x.dtype)
        self.last_little_probs = probs

        # Top-k + 组内归一化，与推理端 top-k streaming 数值上完全等价
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        dense_w = torch.zeros_like(probs).scatter_(-1, topi, topv)
        self.last_little_topk_w = dense_w

        # 稠密算全部 E 个专家再按 top-k 权重掩码：
        #   h = x @ A_flat.T          (N, E*r)
        #   y = (h * w) @ B_flat      (N, D)
        # 相比 "算 E 次再筛 k 个"，这个写法不物化 (N, E, D) 的巨大激活，
        # 而且每个专家都参与计算，梯度不会被 top-k 饿死。
        a_flat = self.lora_A.reshape(-1, dim)  # (E*r, D) —— 连续 view，零拷贝
        b_flat = self.lora_B.reshape(-1, dim)  # (E*r, D)
        h = torch.matmul(x.reshape(n, dim), a_flat.t())  # (N, E*r)
        w_expanded = dense_w.reshape(n, -1, 1).expand(n, -1, self.lora_A.shape[1])
        h = h * w_expanded.reshape(n, -1)
        lora_out = torch.matmul(h, b_flat).reshape(bsz, seqlen, dim)
        lora_out = lora_out * self.scaling

        return big_out + self.gamma * lora_out


# ----------------------------------------------------------------------
# 3. 训练主流程（分级微调，绝不电击）
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("🚀 启动【双大核抢救工程】微温慢火特训 (NVIDIA RTX 5090 D)")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 dtype=dtype,
                                                 device_map="cuda:0")

    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)

    print(f"[*] 正在植入双大核架构与 {NUM_EXPERTS} 微专家...")
    for layer in model.model.layers:
        layer.mlp = DualBigResurrectWrapper(layer.mlp,
                                            hidden_dim,
                                            device="cuda:0",
                                            dtype=dtype)

    # 重点：分级参数组
    big_sci_params = []
    router_params = []
    lora_params = []

    for layer in model.model.layers:
        big_sci_params.extend(
            [p for p in layer.mlp.big_sci.parameters() if p.requires_grad])
        router_params.extend(
            [p for p in layer.mlp.router_big.parameters() if p.requires_grad])
        router_params.extend([
            p for p in layer.mlp.router_little.parameters() if p.requires_grad
        ])
        lora_params.extend([layer.mlp.lora_A, layer.mlp.lora_B])

    print("[✔] 参数隔离完成:")
    print(f"    - 理科大核参数: {sum(p.numel() for p in big_sci_params)/1e6:.1f} M")
    print(f"    - 调度器参数:   {sum(p.numel() for p in router_params)/1e6:.2f} M")
    print(f"    - 微专家参数:   {sum(p.numel() for p in lora_params)/1e6:.1f} M "
          f"({NUM_EXPERTS} x {model.model.layers.__len__()} 层, 全部可训练)")

    dataset = DualContrastDataset("dual_contrast_data.jsonl", tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=MICRO_BATCH,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)

    # 核心救命秘籍：大核 2e-5 微火慢炖，小部件 3e-4~5e-4 快速学习
    optimizer = torch.optim.AdamW([
        {
            "params": big_sci_params,
            "lr": 2e-5,
            "weight_decay": 0.01
        },
        {
            "params": router_params,
            "lr": 3e-4,
            "weight_decay": 0.01
        },
        {
            "params": lora_params,
            "lr": 5e-4,
            "weight_decay": 0.01
        },
    ])
    criterion_router = nn.CrossEntropyLoss()

    total_steps = len(dataloader) // GRAD_ACCUM_STEPS
    print(f"\n[+] 开始慢火细炖 (总步数: {total_steps})...")
    model.train()

    start_time = time.time()
    optimizer.zero_grad()
    running = 0.0
    running_n = 0
    last_report = 0.0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
        labels = batch["labels"].to("cuda:0", non_blocking=True)
        big_target = batch["big_target"].to("cuda:0", non_blocking=True)
        group_target = batch["group_target"].to("cuda:0", non_blocking=True)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        lm_loss = outputs.loss

        # 核心：利用 attention_mask 在整条文本上全量考核调度器
        mask = (attention_mask == 1)
        target_big_seq = big_target.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]
        target_group_seq = group_target.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]

        big_router_loss = 0.0
        little_router_loss = 0.0
        load_balance_loss = 0.0

        for layer in model.model.layers:
            m = layer.mlp

            big_router_loss += criterion_router(m.last_router_big_logits[mask],
                                                target_big_seq)

            # 小核监督改成「分组级」而非「专家级」：
            # 组内 logsumexp 池化后再做 CE，路由器只需保证正确专家组排第一，
            # 组内分工交给 LM loss + 负载均衡项，避免塌缩到单个专家。
            # 注意减去 log(size)：各组大小不等（8/8/16），若用裸 logsumexp
            # 天然偏向大组，写作组会被系统性高估。
            little_logits = m.last_router_little_logits.float()
            group_logits = torch.stack([
                torch.logsumexp(little_logits[..., lo:hi], dim=-1) -
                math.log(hi - lo) for lo, hi in GROUP_BOUNDS
            ],
                                       dim=-1)  # (B, S, NUM_GROUPS)
            little_router_loss += criterion_router(group_logits[mask],
                                                   target_group_seq)

            # Switch Transformer 式负载均衡，防止 32 个专家饿死
            probs = m.last_little_probs[mask].float()
            topk_w = m.last_little_topk_w[mask].float()
            frac = topk_w.mean(dim=0)  # 专家实际承接的 token 比例
            mean_p = probs.mean(dim=0)  # 路由器给出的平均概率
            load_balance_loss += NUM_EXPERTS * torch.sum(frac * mean_p)

        total_loss = (lm_loss + ROUTER_AUX_WEIGHT *
                      (big_router_loss / num_layers) + ROUTER_AUX_WEIGHT *
                      (little_router_loss / num_layers) + LOAD_BALANCE_WEIGHT *
                      (load_balance_loss / num_layers)) / GRAD_ACCUM_STEPS
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            global_step = (step + 1) // GRAD_ACCUM_STEPS
            running += total_loss.item() * GRAD_ACCUM_STEPS
            running_n += 1
            now = time.time()
            # 头 10 步高频反馈，之后每 25 步或每 30s 汇报一次
            if (global_step <= 10 or global_step % 25 == 0
                    or global_step == total_steps
                    or now - last_report > 30):
                last_report = now
                elapsed = time.time() - start_time
                current_loss = running / max(running_n, 1)
                running, running_n = 0.0, 0
                speed = ((step + 1) * MICRO_BATCH) / elapsed
                print(
                    f"    [Step {global_step:03d}/{total_steps}] Loss: {current_loss:.4f} | Speed: {speed:.1f} samples/s | 耗时: {elapsed:.1f}s",
                    flush=True)

    print(
        f"\n[✔] 双大核慢炖抢救特训完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟")

    save_path = "dual_big_resurrect_weights.pt"
    print(f"[*] 正在保存复活后的权重至 {save_path}...")

    state_to_save = {}
    for i, layer in enumerate(model.model.layers):
        state_to_save[f"layer_{i}_big_sci"] = layer.mlp.big_sci.state_dict()
        state_to_save[
            f"layer_{i}_router_big"] = layer.mlp.router_big.state_dict()
        state_to_save[
            f"layer_{i}_router_little"] = layer.mlp.router_little.state_dict()
        state_to_save[f"layer_{i}_lora_A"] = layer.mlp.lora_A.detach().cpu()
        state_to_save[f"layer_{i}_lora_B"] = layer.mlp.lora_B.detach().cpu()

    torch.save(state_to_save, save_path)
    print(f"[✔] 权重成功保存为 {save_path}！")


if __name__ == "__main__":
    main()
