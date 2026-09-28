from collections import Counter
import copy
import os
from threading import Thread
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


# ----------------------------------------------------------------------
# 1. 结构与推理封装
# ----------------------------------------------------------------------
class LoRAMicroExpert(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 rank: int = 16,
                 lora_alpha: float = 16.0,
                 dtype=torch.bfloat16):
        super().__init__()
        self.scaling = lora_alpha / rank
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False, dtype=dtype)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class DualBigResurrectInferenceWrapper(nn.Module):

    def __init__(self,
                 original_mlp: nn.Module,
                 hidden_dim: int,
                 rank: int = 16,
                 num_experts: int = 32,
                 top_k: int = 8,
                 device: str = "cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.top_k = top_k

        self.big_arts = original_mlp.to(device)
        self.big_sci = copy.deepcopy(original_mlp).to(device)

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

        self.lora_pool_cpu = nn.ModuleList([
            LoRAMicroExpert(hidden_dim, rank=rank, dtype=dtype).to("cpu")
            for _ in range(num_experts)
        ])
        for p in self.lora_pool_cpu.parameters():
            p.data = p.data.pin_memory()

        self.transfer_stream = torch.cuda.Stream(device=device)

        self.token_expert_counter = Counter()
        self.total_arts_weight = 0.0
        self.total_sci_weight = 0.0

    def reset_stats(self):
        self.token_expert_counter.clear()
        self.total_arts_weight = 0.0
        self.total_sci_weight = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_token = x[:, -1:, :]

        router_big_logits = self.router_big(current_token)
        weights_big = torch.softmax(router_big_logits, dim=-1)

        self.total_arts_weight += weights_big[0, 0, 0].item()
        self.total_sci_weight += weights_big[0, 0, 1].item()

        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] *
                                                        sci_out)

        router_little_logits = self.router_little(current_token)
        topk_scores, topk_indices = torch.topk(router_little_logits,
                                               k=self.top_k,
                                               dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        selected_ids = topk_indices[0, -1].tolist()
        weights_little = topk_probs[0, -1]

        for eid in selected_ids:
            self.token_expert_counter[eid] += 1

        lora_out = torch.zeros_like(big_out)
        for weight, expert_idx in zip(weights_little, selected_ids):
            with torch.cuda.stream(self.transfer_stream):
                expert_gpu = self.lora_pool_cpu[expert_idx].to(
                    self.device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            lora_out = lora_out + (weight * expert_gpu(x))

        return big_out + (lora_out * 0.3)


# ----------------------------------------------------------------------
# 2. 遥测透析
# ----------------------------------------------------------------------
def show_dual_brain_dashboard(model):
    total_arts = sum(layer.mlp.total_arts_weight
                     for layer in model.model.layers)
    total_sci = sum(layer.mlp.total_sci_weight for layer in model.model.layers)
    all_big = total_arts + total_sci

    arts_pct = (total_arts / all_big * 100) if all_big > 0 else 50
    sci_pct = (total_sci / all_big * 100) if all_big > 0 else 50

    total_counter = Counter()
    for layer in model.model.layers:
        total_counter.update(layer.mlp.token_expert_counter)

    code_calls = sum(total_counter[i] for i in range(0, 8))
    math_calls = sum(total_counter[i] for i in range(8, 16))
    writing_calls = sum(total_counter[i] for i in range(16, 32))
    all_little = code_calls + math_calls + writing_calls

    print("\n" + "═" * 70)
    print("🧠【双大核能量分配 (Tier-1 GPU Resident)】:")
    print(
        f"   🏛️  文科原版大核: {arts_pct:5.1f}% [{'█' * int(arts_pct // 5):<20}] (负责语言通顺与常识底座)"
    )
    print(
        f"   🔬 理科特训大核: {sci_pct:5.1f}% [{'█' * int(sci_pct // 5):<20}] (负责公式、算法与逻辑演算)"
    )
    print("─" * 70)
    if all_little > 0:
        c_pct = (code_calls / all_little) * 100
        m_pct = (math_calls / all_little) * 100
        w_pct = (writing_calls / all_little) * 100
        print("🧩【微专家协同分布 (Tier-2 RAM Streaming)】:")
        print(f"   💻 代码微专家:   {c_pct:5.1f}% ({code_calls:,} 次调用)")
        print(f"   🧮 数学微专家:   {m_pct:5.1f}% ({math_calls:,} 次调用)")
        print(f"   ✍️  写作微专家:   {w_pct:5.1f}% ({writing_calls:,} 次调用)")
    print("═" * 70)


# ----------------------------------------------------------------------
# 3. 对话循环
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    weights_path = "dual_big_resurrect_weights.pt"

    assert os.path.exists(
        weights_path), f"权重文件 {weights_path} 不存在，请确保训练已经执行！"

    print("=" * 70)
    print("🚀 正在装载【复活版·文理双大核】智能终端...")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    for layer in model.model.layers:
        layer.mlp = DualBigResurrectInferenceWrapper(layer.mlp,
                                                     hidden_dim,
                                                     rank=16,
                                                     num_experts=32,
                                                     top_k=8,
                                                     device="cuda:0",
                                                     dtype=dtype)

    print(f"[*] 正在挂载复活成功的双大核与微专家权重...")
    saved_weights = torch.load(weights_path, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.big_sci.load_state_dict(saved_weights[f"layer_{i}_big_sci"])
        layer.mlp.router_big.load_state_dict(
            saved_weights[f"layer_{i}_router_big"])
        layer.mlp.router_little.load_state_dict(
            saved_weights[f"layer_{i}_router_little"])
        layer.mlp.lora_pool_cpu.load_state_dict(
            saved_weights[f"layer_{i}_loras"])

    print("\n✅ 双脑已满血复活！随时可以提问。")
    print("👉 提示：输入 'clear' 重置记忆，输入 'exit' 退出\n")

    messages = [{
        "role":
        "system",
        "content":
        "You are a helpful, precise, and thoughtful assistant."
    }]

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("再见！")
            break
        if user_input.lower() == "clear":
            messages = [{
                "role":
                "system",
                "content":
                "You are a helpful, precise, and thoughtful assistant."
            }]
            print("🧹 对话记忆已清空。")
            continue

        for layer in model.model.layers:
            layer.mlp.reset_stats()

        messages.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generation_kwargs = dict(**inputs,
                                 streamer=streamer,
                                 max_new_tokens=600,
                                 do_sample=True,
                                 temperature=0.7,
                                 top_p=0.9,
                                 repetition_penalty=1.15,
                                 eos_token_id=eos_token_ids)

        print("\n🤖 Assistant: ", end="", flush=True)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        t0 = time.perf_counter()
        thread.start()

        accumulated_text = ""
        try:
            for text_chunk in streamer:
                print(text_chunk, end="", flush=True)
                accumulated_text += text_chunk
        except KeyboardInterrupt:
            print("\n[中断]")

        thread.join()
        elapsed_sec = time.perf_counter() - t0

        gen_tokens_count = len(
            tokenizer.encode(accumulated_text, add_special_tokens=False))
        speed = gen_tokens_count / elapsed_sec if elapsed_sec > 0 else 0
        print(
            f"\n\n⚡ 速度: {speed:.1f} tokens/s (共 {gen_tokens_count} 字, 耗时 {elapsed_sec*1000:.0f} ms)"
        )

        show_dual_brain_dashboard(model)

        messages.append({"role": "assistant", "content": accumulated_text})
        if len(messages) > 7:
            messages = [messages[0]] + messages[-6:]


if __name__ == "__main__":
    main()
