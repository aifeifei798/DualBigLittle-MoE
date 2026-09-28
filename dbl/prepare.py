"""语料装配：文理 1:1 对抗特训语料。

相对原脚本修掉的问题：
  * 原脚本在**模块顶层**执行，没有 ``main()`` 与 ``if __name__`` 保护，
    导致无法被测试 import
  * 产出的 schema（``little_group``）与仓库里既有语料（``little_target``）
    已经漂移，Dataset 的主路径成了死代码
  * gsm8k 语料 100% 带计算器注解 ``#### N``，直接进了训练目标
  * 超长样本被 Dataset 静默截断，训练时白白占位却几乎无梯度信号
  * 没有 train/val 切分，README 的困惑度无法称为泛化指标
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from .config import Config
from .groups import DOMAIN_TO_GROUP

log = logging.getLogger("dbl.prepare")

#: gsm8k 的计算器注解形如 "#### 72"，取最后一个作为答案
GSM8K_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def clean_gsm8k(answer: str) -> str:
    """剥掉 gsm8k 的 ``#### N`` 计算器注解。

    保留推理过程，只把**最后一个**注解（标准答案）改写成自然语言收尾，
    避免模型学会输出裸数字标记。答案里可能提前出现 ``####``（如中间步骤），
    因此必须取最后一个而非第一个。
    """
    matches = list(GSM8K_ANSWER_RE.finditer(answer))
    if not matches:
        return answer.strip()
    last = matches[-1]
    # 正文中可能残留更早的注解，一并清除，避免模型学会输出裸标记
    body = GSM8K_ANSWER_RE.sub("", answer).strip()
    final = last.group(1).strip().rstrip(".")
    return f"{body}\nThe answer is {final}."


def build_code_rows(ds, limit: int) -> list[dict]:
    rows = []
    for item in ds:
        if len(rows) >= limit:
            break
        prompt = item["instruction"] + (f"\n{item['input']}" if item.get("input") else "")
        response = (item.get("output") or "").strip()
        if not response:
            continue
        rows.append(
            {
                "big_target": 1,
                "little_group": DOMAIN_TO_GROUP["Code"],
                "domain": "Code",
                "prompt": prompt.strip(),
                "response": response,
            }
        )
    return rows


def build_math_rows(ds, limit: int) -> list[dict]:
    rows = []
    for item in ds:
        if len(rows) >= limit:
            break
        response = clean_gsm8k(item["answer"])
        if not response:
            continue
        rows.append(
            {
                "big_target": 1,
                "little_group": DOMAIN_TO_GROUP["Math"],
                "domain": "Math",
                "prompt": item["question"].strip(),
                "response": response,
            }
        )
    return rows


def build_arts_rows(ds, limit: int) -> list[dict]:
    """从 no_robots 取单轮 assistant 回复。

    过滤掉 role != assistant 的样本，否则会把用户侧文本当成模型应当
    生成的内容来监督。
    """
    rows = []
    for item in ds:
        if len(rows) >= limit:
            break
        messages = item.get("messages") or []
        if len(messages) < 2 or messages[1].get("role") != "assistant":
            continue
        prompt = (messages[0].get("content") or "").strip()
        response = messages[1].get("content", "").strip()
        if not prompt or not response:
            continue
        rows.append(
            {
                "big_target": 0,
                "little_group": DOMAIN_TO_GROUP["Arts"],
                "domain": "Arts",
                "prompt": prompt,
                "response": response,
            }
        )
    return rows


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple[str, str]] = set()
    out, dropped = [], 0
    for r in rows:
        key = (r["prompt"], r["response"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dropped


def length_filter(rows: list[dict], tokenizer, max_length: int) -> tuple[list[dict], dict]:
    """丢弃 prompt 本身就超过 max_length 的样本。

    这类样本在 Dataset 里会被截断到只剩 prompt，response 一字不剩，
    等于给了一个恒为 -100 的标签，纯粹浪费算力。
    """
    kept, stats = [], Counter()
    for r in rows:
        n_prompt = len(
            tokenizer(
                f"<|im_start|>user\n{r['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n",
                add_special_tokens=False,
            )["input_ids"]
        )
        if n_prompt >= max_length - 8:      # 至少给 response 留 8 token
            stats[r["domain"] + ":prompt_too_long"] += 1
            continue
        kept.append(r)
    return kept, dict(stats)


def split_by_domain(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """按 domain 分层切分，保证 train/val 的领域配比一致。"""
    import random

    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    train, val = [], []
    for items in by_domain.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dual_contrast_data.jsonl")
    ap.add_argument("--val-out", default="dual_contrast_val.jsonl")
    ap.add_argument("--n-code", type=int, default=2000)
    ap.add_argument("--n-math", type=int, default=2000)
    ap.add_argument("--n-arts", type=int, default=4000)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="从 config json 读取 max_length 等")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config.load(args.config) if args.config else Config()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)

    log.info("拉取语料 Code=%d Math=%d Arts=%d", args.n_code, args.n_math, args.n_arts)
    rows: list[dict] = []
    rows += build_code_rows(
        load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                     split=f"train[:{args.n_code}]"),
        args.n_code,
    )
    rows += build_math_rows(
        load_dataset("openai/gsm8k", "main", split=f"train[:{args.n_math}]"),
        args.n_math,
    )
    rows += build_arts_rows(
        load_dataset("HuggingFaceH4/no_robots", split=f"train[:{args.n_arts}]"),
        args.n_arts,
    )
    log.info("原始 %d 条", len(rows))

    rows, dup = dedupe(rows)
    log.info("去重丢弃 %d 条，剩 %d 条", dup, len(rows))

    rows, dropped = length_filter(rows, tok, args.max_length)
    if dropped:
        log.info("按长度丢弃 %d 条: %s", sum(dropped.values()), dropped)

    train_rows, val_rows = split_by_domain(rows, args.val_ratio, args.seed)
    write_jsonl(train_rows, Path(args.out))
    write_jsonl(val_rows, Path(args.val_out))

    log.info("train %d 条 -> %s", len(train_rows), args.out)
    log.info("val   %d 条 -> %s", len(val_rows), args.val_out)
    log.info("领域分布: %s", dict(Counter(r["domain"] for r in train_rows)))
    log.info("文理配比: %s", dict(Counter(r["big_target"] for r in train_rows)))


if __name__ == "__main__":
    main()
