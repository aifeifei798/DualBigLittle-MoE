import copy
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# 1. 数据集加载
# ----------------------------------------------------------------------
class DualContrastDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
        tokens = self.tokenizer(text,
                                max_length=self.max_length,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt")

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids":
            input_ids,
            "attention_mask":
            attention_mask,
            "labels":
            labels,
            "big_target":
            torch.tensor(item["big_target"], dtype=torch.long),
            "little_target":
            torch.tensor(item["little_target"], dtype=torch.long)
        }


# ----------------------------------------------------------------------
# 2. 微专家结构
# ----------------------------------------------------------------------
class TrainableLoRAMicroExpert(nn.Module):

    def __init__(self,
                 hidden_dim,
                 rank=16,
                 lora_alpha=16.0,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim,
                                rank,
                                bias=False,
                                device=device,
                                dtype=dtype)
        self.lora_B = nn.Linear(rank,
                                hidden_dim,
                                bias=False,
                                device=device,
                                dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


# ----------------------------------------------------------------------
# 3. 架构定义 (双大核安全混音)
# ----------------------------------------------------------------------
class DualBigResurrectWrapper(nn.Module):

    def __init__(self,
                 original_mlp,
                 hidden_dim,
                 rank=16,
                 num_experts=32,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.num_experts = num_experts

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
        self.last_router_big_logits = None

        # 32 微专家池 + 调度器
        self.router_little = nn.Linear(hidden_dim,
                                       num_experts,
                                       bias=False,
                                       device=device,
                                       dtype=dtype)
        self.last_router_little_logits = None
        self.lora_pool = nn.ModuleList([
            TrainableLoRAMicroExpert(hidden_dim,
                                     rank=rank,
                                     device=device,
                                     dtype=dtype) for _ in range(num_experts)
        ])

        self.current_little_target = None

    def forward(self, x):
        router_big_logits = self.router_big(x)
        self.last_router_big_logits = router_big_logits
        weights_big = torch.softmax(router_big_logits, dim=-1)

        with torch.no_grad():
            arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)

        big_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] *
                                                        sci_out)

        router_little_logits = self.router_little(x)
        self.last_router_little_logits = router_little_logits

        # 根据班长目标定点训练小核
        batch_size = x.size(0)
        lora_outs = []
        for b in range(batch_size):
            exp_idx = self.current_little_target[b].item()
            lora_outs.append(self.lora_pool[exp_idx](x[b:b + 1]))
        lora_out = torch.cat(lora_outs, dim=0)

        return big_out + 0.3 * lora_out


# ----------------------------------------------------------------------
# 4. 训练主流程 (分级微调，绝不电击)
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    print("=" * 70)
    print("🚀 启动【双大核抢救工程】微温慢火特训 (NVIDIA RTX 5090 D)")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")

    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)

    print(f"[*] 正在植入双大核架构与 32 微专家...")
    for layer in model.model.layers:
        layer.mlp = DualBigResurrectWrapper(layer.mlp,
                                            hidden_dim,
                                            rank=16,
                                            num_experts=32,
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
        lora_params.extend(
            [p for p in layer.mlp.lora_pool.parameters() if p.requires_grad])

    print(f"[✔] 参数隔离完成:")
    print(f"    - 理科大核参数: {sum(p.numel() for p in big_sci_params)/1e6:.1f} M")
    print(f"    - 调度器参数:   {sum(p.numel() for p in router_params)/1e6:.2f} M")
    print(f"    - 微专家参数:   {sum(p.numel() for p in lora_params)/1e6:.1f} M")

    MICRO_BATCH = 4
    GRAD_ACCUM_STEPS = 4
    MAX_LENGTH = 512

    dataset = DualContrastDataset("dual_contrast_data.jsonl",
                                  tokenizer,
                                  max_length=MAX_LENGTH)
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

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
        labels = batch["labels"].to("cuda:0", non_blocking=True)
        big_target = batch["big_target"].to("cuda:0", non_blocking=True)
        little_target = batch["little_target"].to("cuda:0", non_blocking=True)

        for layer in model.model.layers:
            layer.mlp.current_little_target = little_target

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        lm_loss = outputs.loss

        # 核心：利用 attention_mask 在整条文本上全量考核调度器
        mask = (attention_mask == 1)
        target_big_seq = big_target.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]
        target_little_seq = little_target.unsqueeze(1).expand(
            -1, MAX_LENGTH)[mask]

        big_router_loss = 0.0
        little_router_loss = 0.0

        for layer in model.model.layers:
            active_logits_big = layer.mlp.last_router_big_logits[mask]
            active_logits_little = layer.mlp.last_router_little_logits[mask]

            big_router_loss += criterion_router(active_logits_big,
                                                target_big_seq)
            little_router_loss += criterion_router(active_logits_little,
                                                   target_little_seq)

        total_loss = (lm_loss + 0.1 * (big_router_loss / num_layers) +
                      0.1 * (little_router_loss / num_layers)) / GRAD_ACCUM_STEPS
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            global_step = (step + 1) // GRAD_ACCUM_STEPS
            if global_step % 25 == 0 or global_step == total_steps:
                elapsed = time.time() - start_time
                current_loss = total_loss.item() * GRAD_ACCUM_STEPS
                speed = ((step + 1) * MICRO_BATCH) / elapsed
                print(
                    f"    [Step {global_step:03d}/{total_steps}] Loss: {current_loss:.4f} | Speed: {speed:.1f} samples/s | 耗时: {elapsed:.1f}s"
                )

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
        state_to_save[
            f"layer_{i}_loras"] = layer.mlp.lora_pool.state_dict()

    torch.save(state_to_save, save_path)
    print(f"[✔] 权重成功保存为 {save_path}！")


if __name__ == "__main__":
    main()
