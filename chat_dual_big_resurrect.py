#!/usr/bin/env python
"""交互对话：加载复活后的双大核权重，PCIe 流式推理。

用法::

    python chat_dual_big_resurrect.py
    python chat_dual_big_resurrect.py --weights runs/best.pt
    python chat_dual_big_resurrect.py --no-telemetry     # 关掉每次回答后的统计

结构超参（专家数、top-k、分组拓扑等）全部从 checkpoint 自带的 config 读取，
不再需要在本文件里维护第二份常量。
"""

from __future__ import annotations

import argparse
import logging
import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from dbl.checkpoint import apply_checkpoint, load_checkpoint
from dbl.config import Config
from dbl.moe import InferMoE, inject_moe
from dbl.runtime import fail_cli, format_report, resolve_device

log = logging.getLogger("dbl.chat")

SYSTEM_PROMPT = "You are a helpful, precise, and thoughtful assistant."
COMMANDS = {"clear", "exit", "quit"}


def build_model(weights_path: str, device: str):
    """构造基座 -> 注入推理模块 -> 装载权重（超参取自 checkpoint）。"""
    payload = load_checkpoint(weights_path, map_location="cpu")
    cfg: Config = payload.get("cfg")
    if cfg is None:
        raise SystemExit(
            f"{weights_path} 是旧格式 checkpoint，不含超参。\n"
            "请用新脚本重新训练，或显式传 --config 指定配置文件。"
        )
    cfg = cfg.replace(device=device)
    log.info("从 checkpoint 读取结构超参: experts=%d top_k=%d gamma=%.2f "
             "groups=%s", cfg.num_experts, cfg.top_k, cfg.gamma,
             cfg.groups.sizes)

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.torch_dtype, device_map=device
    )
    inject_moe(model, cfg, mode="infer")
    info = apply_checkpoint(
        [layer.mlp for layer in model.model.layers], payload
    )
    # Tier-3 专家池搬到 pinned 主机内存
    for layer in model.model.layers:
        m = layer.mlp
        m.load_expert_pool(m.lora_A.data, m.lora_B.data)
    model.eval()
    return model, tok, cfg, info


@torch.no_grad()
def dashboard(modules: list[InferMoE], groups) -> None:
    """一次性取回全部遥测统计并打印（避免每层各自同步）。"""
    arts = torch.stack([m.telemetry["arts_weight"] for m in modules]).sum().item()
    sci = torch.stack([m.telemetry["sci_weight"] for m in modules]).sum().item()
    total = arts + sci
    arts_pct = arts / total * 100 if total > 0 else 50.0
    sci_pct = sci / total * 100 if total > 0 else 50.0

    counts = torch.stack(
        [m.telemetry["expert_calls"] for m in modules]
    ).sum(0).cpu()
    shares = groups.expert_share(counts.tolist())
    all_little = sum(counts.tolist())

    print("\n" + "═" * 68)
    print("【双大核能量分配】")
    print(f"   文科锚核(Tier-1): {arts_pct:5.1f}% "
          f"[{'█' * int(arts_pct // 5):<20}]")
    print(f"   理科孪生核(Tier-2): {sci_pct:5.1f}% "
          f"[{'█' * int(sci_pct // 5):<20}]")
    if all_little > 0:
        print("─" * 68)
        print("【微专家调用分布 (Tier-3 主机内存流式)】")
        labels = groups.labels()
        for i, (lo, hi) in enumerate(groups.bounds):
            size = hi - lo
            fair = size / groups.num_experts * 100
            label = labels[i] if i < len(labels) else f"组{i}"
            pct = shares[i] * 100
            print(f"   {label:>4} ({size:2d} 专家): {pct:5.1f}%  "
                  f"均衡度 {pct / fair if fair > 0 else 0:.2f}")
    print("═" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="dual_big_resurrect_weights.pt")
    ap.add_argument("--device", default="auto",
                    help="auto / cpu / cuda / cuda:N，默认自动探测")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-history", type=int, default=6)
    ap.add_argument("--no-telemetry", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        args.device = resolve_device(args.device, allow_cpu=False)
    except (RuntimeError, ValueError) as exc:
        fail_cli(exc, "chat_dual_big_resurrect.py")
        return
    log.info("对话环境\n%s", format_report())

    model, tok, cfg, info = build_model(args.weights, args.device)
    modules = [layer.mlp for layer in model.model.layers]
    log.info("权重装载完成: %s", info)
    if info.get("quant_rel_error") is not None:
        log.info("int8 量化相对误差: %.2e", info["quant_rel_error"])

    eos = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id:
        eos.append(im_end)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\n双脑已就绪。输入 'clear' 重置记忆，'exit' 退出\n")

    while True:
        try:
            user = input("\n👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break
        if not user:
            continue
        low = user.lower()
        if low in ("exit", "quit"):
            print("再见！")
            break
        if low == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("对话记忆已清空。")
            continue

        for m in modules:
            m.reset_telemetry()

        messages.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt, return_tensors="pt").to(args.device)

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos,
        )

        print("\n🤖 Assistant: ", end="", flush=True)
        t0 = time.perf_counter()
        th = Thread(target=model.generate, kwargs=kwargs)
        th.start()
        text = ""
        try:
            for chunk in streamer:
                print(chunk, end="", flush=True)
                text += chunk
        except KeyboardInterrupt:
            print("\n[已中断]")
        th.join()
        dt = time.perf_counter() - t0

        n_tok = len(tok.encode(text, add_special_tokens=False))
        print(f"\n\n⚡ {n_tok / dt if dt > 0 else 0:.1f} tokens/s  ({dt * 1000:.0f} ms)")

        if not args.no_telemetry:
            dashboard(modules, cfg.groups)

        messages.append({"role": "assistant", "content": text})
        if len(messages) > args.max_history + 1:
            messages = [messages[0]] + messages[-(args.max_history):]


if __name__ == "__main__":
    main()
