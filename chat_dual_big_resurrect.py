from collections import Counter
import copy
import os
import time
from threading import Thread

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ----------------------------------------------------------------------
# 0. 全局超参（必须与 train_dual_big_resurrect.py 保持一致）
# ----------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-0.6B"
TOP_K = 8
GAMMA = 0.3
LORA_RANK = 16
LORA_ALPHA = 16.0
NUM_EXPERTS = 32
NUM_STAGING_BUFFERS = 2  # 双缓冲：拷贝与计算真正重叠

# 遥测分组，必须与 train_dual_big_resurrect.py 的 GROUP_BOUNDS 一致
GROUP_BOUNDS = [(0, 8), (8, 16), (16, 32)]
GROUP_LABELS = ["💻 代码", "🧮 数学", "✍️  写作"]


# ----------------------------------------------------------------------
# 1. 推理封装：主机 pinned RAM 专家池 + 真正的双缓冲 DMA 流式加载
# ----------------------------------------------------------------------
class DualBigResurrectInferenceWrapper(nn.Module):
    """Tier-1/Tier-2 常驻显存，Tier-3 微专家留在 pinned 主机内存按需流式搬运。

    关键点
    ------
    * staging buffer 是**常驻**的（不走 caching allocator），因此不存在
      "在 transfer_stream 上分配、在 default stream 上使用、随即被回收"
      的竞态（也就不需要 record_stream）。
    * top-k 个专家的 H2D 拷贝一次性全部发到 transfer_stream，**发完再等**，
      配合双缓冲与上一拍的完成事件，拷贝和计算真正重叠。
    * 遥测统计全部留在 GPU 上累加，每层每 token 只做 **1 次** D2H 同步
      （且那次同步本来就是拿专家下标给主机内存用，顺路把权重带回来）。
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
                 dtype=torch.bfloat16,
                 num_staging_buffers=NUM_STAGING_BUFFERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gamma = gamma
        self.scaling = lora_alpha / rank
        self.device = device
        self.dtype = dtype

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

        # Tier-3：主机端专家池（pinned，形状 (E, r, D)，每专家 2*r*D*2B = 64KB）
        self.host_lora_A = torch.zeros(num_experts,
                                       rank,
                                       hidden_dim,
                                       dtype=dtype)
        self.host_lora_B = torch.zeros(num_experts,
                                       rank,
                                       hidden_dim,
                                       dtype=dtype)
        self.host_pinned = False

        # 常驻 staging buffer：k 个槽位 x 2 组，做 ping-pong
        self.staging_A = [
            torch.empty(self.top_k, rank, hidden_dim, device=device,
                        dtype=dtype) for _ in range(num_staging_buffers)
        ]
        self.staging_B = [
            torch.empty(self.top_k, rank, hidden_dim, device=device,
                        dtype=dtype) for _ in range(num_staging_buffers)
        ]
        self.staging_events = [torch.cuda.Event() for _ in range(num_staging_buffers)]
        cur = torch.cuda.current_stream()
        for ev in self.staging_events:  # 首拍不要阻塞
            ev.record(cur)
        self._staging_idx = 0

        self.transfer_stream = torch.cuda.Stream(device=device)

        # --- 遥测累加器：全部驻留 GPU，避免每层每 token 的 .item() 同步 ---
        self.arts_acc = torch.zeros((), device=device, dtype=torch.float32)
        self.sci_acc = torch.zeros((), device=device, dtype=torch.float32)
        self.expert_call_counts = torch.zeros(num_experts,
                                              device=device,
                                              dtype=torch.long)

    # ------------------------------------------------------------------
    def load_expert_pool(self, lora_a: torch.Tensor, lora_b: torch.Tensor):
        """把训练好的堆叠权重拷到 pinned 主机内存。

        dtype 跟随模型自身，不做隐式降级——否则 fp32/fp16 推理时专家池会被
        悄悄压成 bf16，训练侧与推理侧产生无法解释的数值偏差。
        """
        expected = (self.num_experts, self.rank, self.hidden_dim)
        assert tuple(lora_a.shape) == expected, f"lora_A 形状应为 {expected}"
        assert tuple(lora_b.shape) == expected, f"lora_B 形状应为 {expected}"
        self.host_lora_A = lora_a.detach().to("cpu", self.dtype).contiguous(
        ).pin_memory()
        self.host_lora_B = lora_b.detach().to("cpu", self.dtype).contiguous(
        ).pin_memory()
        self.host_pinned = True

    def reset_stats(self):
        self.arts_acc.zero_()
        self.sci_acc.zero_()
        self.expert_call_counts.zero_()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.host_pinned, "Tier-3 专家池尚未 pin 到主机内存！"
        bsz, seqlen, dim = x.shape
        n = bsz * seqlen

        # --- 大核：逐位置路由（与训练完全一致；decode 时等价于只看最后 token）---
        router_big_logits = self.router_big(x)
        weights_big = torch.softmax(router_big_logits.float(),
                                    dim=-1).to(x.dtype)
        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (weights_big[..., 0:1] * arts_out) + (weights_big[..., 1:2] *
                                                        sci_out)

        # --- 小核：prefill 用最后一个 token 决定专家集合（一个 prompt 只搬一轮）---
        last_tok = x[:, -1:, :]
        router_little_logits = self.router_little(last_tok)
        probs = torch.softmax(router_little_logits.float(), dim=-1).to(x.dtype)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)

        # 遥测：GPU 侧累加，零同步
        self.arts_acc += weights_big[0, -1, 0].float()
        self.sci_acc += weights_big[0, -1, 1].float()
        self.expert_call_counts.index_add_(
            0, topi[0, -1],
            torch.ones(self.top_k, device=self.device, dtype=torch.long))

        # 唯一的 D2H 同步：专家下标 + 归一化权重，一次打包取回
        info = torch.stack([topi[0, -1].to(x.dtype),
                            topv[0, -1]]).to("cpu")
        info_list = info.tolist()
        selected_ids = [int(v) for v in info_list[0]]
        lora_weights = [float(v) for v in info_list[1]]

        # --- Tier-3：一次性把 k 个专家发到 transfer_stream，再统一等待 ---
        slot = self._staging_idx
        self._staging_idx = (self._staging_idx + 1) % len(self.staging_A)

        # 等待两拍之前读这块 buffer 的计算结束，保证不会覆写正在被用的数据
        self.transfer_stream.wait_event(self.staging_events[slot])
        with torch.cuda.stream(self.transfer_stream):
            for j, eid in enumerate(selected_ids):
                self.staging_A[slot][j].copy_(self.host_lora_A[eid],
                                              non_blocking=True)
                self.staging_B[slot][j].copy_(self.host_lora_B[eid],
                                              non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # --- 融合：k 次 (N,r)x(r,D) 小 GEMM，用 addmm_ 原地累加 ---
        x2d = x.reshape(n, dim)
        a_buf = self.staging_A[slot]
        b_buf = self.staging_B[slot]
        lora_flat = torch.zeros(n, dim, device=self.device, dtype=x.dtype)
        for j, w in enumerate(lora_weights):
            h = torch.matmul(x2d, a_buf[j].t())  # (N, r)
            lora_flat.addmm_(h, b_buf[j], alpha=w)  # lora_flat += w * h @ B

        # 标记本拍读取该 buffer 的计算，供两拍后的拷贝等待
        self.staging_events[slot].record(torch.cuda.current_stream())

        lora_out = (lora_flat.reshape(bsz, seqlen, dim) * self.scaling)
        return big_out + self.gamma * lora_out


# ----------------------------------------------------------------------
# 2. 遥测透析
# ----------------------------------------------------------------------
def show_dual_brain_dashboard(model):
    # 只在这里做 GPU -> CPU，一次性取回全部统计量
    layers = list(model.model.layers)
    total_arts = torch.stack(
        [layer.mlp.arts_acc for layer in layers]).sum().item()
    total_sci = torch.stack([layer.mlp.sci_acc for layer in layers]).sum().item()
    all_big = total_arts + total_sci

    arts_pct = (total_arts / all_big * 100) if all_big > 0 else 50.0
    sci_pct = (total_sci / all_big * 100) if all_big > 0 else 50.0

    total_counter = torch.stack(
        [layer.mlp.expert_call_counts for layer in layers]).sum(0).cpu()
    counts = total_counter.tolist()

    group_calls = [sum(counts[lo:hi]) for lo, hi in GROUP_BOUNDS]
    all_little = sum(group_calls)

    print("\n" + "═" * 70)
    print("🧠【双大核能量分配 (Tier-1 & Tier-2 GPU Resident)】:")
    print(
        f"   🏛️  文科原版大核: {arts_pct:5.1f}% [{'█' * int(arts_pct // 5):<20}] (负责语言通顺与常识底座)"
    )
    print(
        f"   🔬 理科特训大核: {sci_pct:5.1f}% [{'█' * int(sci_pct // 5):<20}] (负责公式、算法与逻辑演算)"
    )
    print("─" * 70)
    if all_little > 0:
        print("🧩【微专家协同分布 (Tier-3 Host RAM Streaming)】:")
        for label, (lo, hi), calls in zip(GROUP_LABELS, GROUP_BOUNDS,
                                          group_calls):
            pct = calls / all_little * 100
            size = hi - lo
            # 均衡度：实际占比 / 按容量应得占比，1.0 = 完全均衡
            fair = size / len(counts) * 100
            print(f"   {label}微专家: {pct:5.1f}% ({calls:,} 次调用, "
                  f"{size} 个专家, 均衡度 {pct/fair:.2f})")
    print("═" * 70)


# ----------------------------------------------------------------------
# 3. 对话循环
# ----------------------------------------------------------------------
def main():
    weights_path = "dual_big_resurrect_weights.pt"

    assert os.path.exists(
        weights_path), f"权重文件 {weights_path} 不存在，请确保训练已经执行！"

    print("=" * 70)
    print("🚀 正在装载【复活版·文理双大核】智能终端...")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_token_ids.append(im_end_id)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size
    model.eval()

    for layer in model.model.layers:
        layer.mlp = DualBigResurrectInferenceWrapper(layer.mlp,
                                                     hidden_dim,
                                                     device="cuda:0",
                                                     dtype=dtype)

    print("[*] 正在挂载复活成功的双大核与微专家权重...")
    saved_weights = torch.load(weights_path,
                               map_location="cpu",
                               weights_only=True)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.big_sci.load_state_dict(saved_weights[f"layer_{i}_big_sci"])
        layer.mlp.router_big.load_state_dict(
            saved_weights[f"layer_{i}_router_big"])
        layer.mlp.router_little.load_state_dict(
            saved_weights[f"layer_{i}_router_little"])
        layer.mlp.load_expert_pool(saved_weights[f"layer_{i}_lora_A"],
                                   saved_weights[f"layer_{i}_lora_B"])
    del saved_weights

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
