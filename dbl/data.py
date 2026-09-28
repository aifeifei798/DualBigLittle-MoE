"""语料加载与编码。

标签约定（与 :mod:`dbl.groups` 的分组拓扑一致）：
    big_target   0 -> 文科大核(Tier 1)，1 -> 理科大核(Tier 2)
    group_target 分组下标，0/1/2 对应 代码/数学/写作

``labels`` 只监督 response 段，prompt 段与 padding 全部置 -100。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import Config

REQUIRED_FIELDS = ("big_target", "prompt", "response")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} JSON 解析失败：{exc}") from exc
    return rows


def resolve_group(item: dict[str, Any], groups) -> int:
    """取分组下标，兼容新旧两种语料 schema。

    优先使用 ``little_group``（当前 schema）；旧语料只有 ``little_target``，
    存的是分组起始下标（0/8/16），需反查所属区间。
    """
    if item.get("little_group") is not None:
        return int(item["little_group"])
    domain = item.get("domain")
    if domain is not None:
        return groups.group_of_domain(domain)
    lt = int(item.get("little_target", 0))
    for g, (lo, hi) in enumerate(groups.bounds):
        if lo <= lt < hi:
            return g
    raise ValueError(f"little_target={lt} 不落在任何分组区间内")


class DualContrastDataset(Dataset):
    """文理 1:1 对抗语料。"""

    def __init__(self, path, tokenizer, cfg: Config):
        self.rows = load_jsonl(path)
        missing = {
            f
            for r in self.rows[:64]
            for f in REQUIRED_FIELDS
            if f not in r
        }
        if missing:
            raise ValueError(
                f"{path} 缺少必需字段 {sorted(missing)}；"
                "请用 scripts/prepare_dual_data.py 重新生成语料"
            )
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.groups = cfg.groups
        self.max_length = cfg.max_length
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer 缺少 pad_token，请先设置 tokenizer.pad_token")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.rows[idx]

        prompt_text = (
            f"<|im_start|>user\n{item['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_ids = self.tokenizer(
            prompt_text + item["response"] + "<|im_end|>\n",
            add_special_tokens=False,
        )["input_ids"]

        truncated = len(full_ids) > self.max_length
        full_ids = full_ids[: self.max_length]

        # 只监督 response：prompt 部分不计 loss
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_prompt + full_ids[n_prompt:]

        pad_len = self.max_length - len(full_ids)
        input_ids = full_ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "big_target": torch.tensor(int(item["big_target"]), dtype=torch.long),
            "group_target": torch.tensor(
                resolve_group(item, self.groups), dtype=torch.long
            ),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
        }
