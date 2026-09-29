"""DualBigLittle-MoE 上手示例。

    python example_usage.py                      # 贪心生成
    python example_usage.py --stream             # 流式输出
    python example_usage.py --telemetry          # 双大核能量分配 + 专家调用分布
    python example_usage.py --model your-org/dualbig-qwen3-0.6b

本文件由 ``export_to_hf.py`` 复制进导出目录，因此是外部用户拿到权重后
第一个该跑的东西 —— 它的失败模式就是外部用户的第一印象。
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 默认为本脚本所在目录，这样从任意 cwd 运行都能找到模型；
# 指向 Hub 仓库时用 --model 或环境变量 DUALBIG_MODEL 覆盖。
MODEL_PATH = os.environ.get("DUALBIG_MODEL", str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL_PATH, help="模型目录或 Hub 仓库 id")
    ap.add_argument("--prompt", default="用一句话解释什么是二分查找。")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--telemetry", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device).eval()

    cfg = model.config
    print(f"模型类型 : {type(model).__name__}")
    print(
        f"专家配置 : {cfg.num_experts} 路 / Top-{cfg.top_k_experts} "
        f"/ rank {cfg.lora_rank} / gamma {cfg.gamma}"
    )
    print(f"专家池   : {cfg.expert_pool_location}")

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(text, return_tensors="pt").to(args.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

    if args.stream:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=model.generate,
            kwargs={**inputs, **gen_kwargs, "streamer": streamer},
        )
        thread.start()
        print("\n助手 > ", end="", flush=True)
        for chunk in streamer:
            print(chunk, end="", flush=True)
        print()
        thread.join()
    else:
        out = model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[-1]:]
        print("\n助手 > " + tok.decode(gen, skip_special_tokens=True))

    if args.telemetry:
        report = model.routing_report()
        if not report:
            return
        print(
            f"\n双大核能量分配: 文科锚核 {report['arts_ratio']:.1%} / "
            f"理科孪生核 {report['sci_ratio']:.1%}"
        )
        for (lo, hi), share, fair in zip(
            report["group_bounds"],
            report["group_shares"],
            report["group_fair_share"],
            strict=True,
        ):
            bar = "#" * int(share * 40)
            print(
                f"  专家 {lo:2d}-{hi - 1:<2d}: {share:6.1%} "
                f"(均衡基准 {fair:.1%}) {bar}"
            )


if __name__ == "__main__":
    main()
