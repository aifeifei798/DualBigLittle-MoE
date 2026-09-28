#!/usr/bin/env python
"""分域评估与可复现性体检。

本脚本在**单一前向遍历**中同时采集四类指标，用于替代 README 中手写的实测表格：

1. 分域困惑度（PPL）    —— Code / Math / Arts，与原生 Qwen3-0.6B baseline 对比
2. 专家健康度          —— 获得非零权重的专家数、完全未被调用的专家数
3. 路由分化            —— 逐层 big_sci（理科大核）权重均值
4. 数值抖动            —— 同一输入重复前向的输出差异

同时支持训练路径（稠密 top-k，保留全部专家梯度）与推理路径
（末 token top-k + pinned 流式搬运）的一致性对比。

用法::

    python eval_ppl.py --out reports/baseline.json

注意：当前语料 **没有** train/val 切分，默认评估的是语料尾部的同分布切片，
属于 in-distribution 指标，不能等同于泛化性能。阶段 C 引入真实切分后需重测。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 预重构阶段：直接从现有脚本导入，保证评估逻辑与被评估对象同源。
from train_dual_big_resurrect import (  # noqa: E402
    GROUP_BOUNDS,
    MODEL_ID,
    DualBigResurrectWrapper,
    DualContrastDataset,
)
from chat_dual_big_resurrect import (  # noqa: E402
    DualBigResurrectInferenceWrapper,
)

DOMAINS = ("Code", "Math", "Arts")


# ----------------------------------------------------------------------
# 采样
# ----------------------------------------------------------------------
def pick_indices(domains: dict[str, list[int]], per_domain: int) -> list[int]:
    """在每个 domain 内等距取样，确定性且覆盖整个分布。"""
    picked: list[int] = []
    for domain in DOMAINS:
        idx = domains[domain]
        if not idx:
            continue
        if len(idx) <= per_domain:
            picked.extend(idx)
        else:
            step = len(idx) / per_domain
            picked.extend(idx[int(i * step)] for i in range(per_domain))
    return sorted(picked)


def read_domains(data_path: str) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    with open(data_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                buckets[json.loads(line).get("domain", "Arts")].append(i)
    return dict(buckets)


# ----------------------------------------------------------------------
# 模型构建
# ----------------------------------------------------------------------
def build_baseline(device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )
    model.eval()
    return model


def _load_weights(weights_path: str) -> dict:
    return torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)


def _inject(model, wrapper_cls, device, dtype, **kwargs):
    hidden = model.config.hidden_size
    for layer in model.model.layers:
        layer.mlp = wrapper_cls(
            layer.mlp, hidden, device=device, dtype=dtype, **kwargs
        )
    return model


def build_dual_train_impl(weights_path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )
    for p in model.parameters():
        p.requires_grad = False
    _inject(model, DualBigResurrectWrapper, device, dtype)

    sd = _load_weights(weights_path)
    for i, layer in enumerate(model.model.layers):
        m = layer.mlp
        m.big_sci.load_state_dict(sd[f"layer_{i}_big_sci"])
        m.router_big.load_state_dict(sd[f"layer_{i}_router_big"])
        m.router_little.load_state_dict(sd[f"layer_{i}_router_little"])
        m.lora_A.data.copy_(sd[f"layer_{i}_lora_A"].to(device, dtype))
        m.lora_B.data.copy_(sd[f"layer_{i}_lora_B"].to(device, dtype))
    del sd
    model.eval()
    return model


def build_dual_infer_impl(weights_path: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )
    _inject(model, DualBigResurrectInferenceWrapper, device, dtype)

    sd = _load_weights(weights_path)
    for i, layer in enumerate(model.model.layers):
        m = layer.mlp
        m.big_sci.load_state_dict(sd[f"layer_{i}_big_sci"])
        m.router_big.load_state_dict(sd[f"layer_{i}_router_big"])
        m.router_little.load_state_dict(sd[f"layer_{i}_router_little"])
        m.load_expert_pool(
            sd[f"layer_{i}_lora_A"], sd[f"layer_{i}_lora_B"]
        )
    del sd
    model.eval()
    return model


# ----------------------------------------------------------------------
# 单次前向采集：PPL + 路由 + 专家
# ----------------------------------------------------------------------
@torch.no_grad()
def analyze(
    model,
    dataset: DualContrastDataset,
    indices: list[int],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """一次遍历收集全部指标。

    PPL 走因果 LM 标准移位，且只统计 label != -100 的 response 位置，
    与训练目标（prompt 掩码）保持一致。
    路由/专家统计走 attention_mask，即全部真实 token —— 与训练时辅助损失的口径相同。
    """
    is_moe = hasattr(model.model.layers[0].mlp, "last_router_big_logits")
    n_layers = len(model.model.layers)
    n_experts = (
        model.model.layers[0].mlp.router_little.out_features if is_moe else 0
    )

    nll_sum: dict[str, float] = defaultdict(float)
    tok_count: dict[str, int] = defaultdict(int)
    dead_labels = 0
    expert_w = (
        torch.zeros(n_layers, n_experts, dtype=torch.float64, device=device)
        if is_moe
        else None
    )
    sci_w = torch.zeros(n_layers, dtype=torch.float64, device=device) if is_moe else None
    sci_tok = 0

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        batch = {
            k: torch.stack([it[k] for it in items]).to(device, non_blocking=True)
            for k in ("input_ids", "attention_mask", "labels")
        }
        domain = dataset.samples[chunk[0]]["domain"]

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = out.logits

        # --- PPL：标准因果移位，只算 response ---
        shift_logits = logits[:, :-1, :]
        shift_labels = batch["labels"][:, 1:]
        sel = shift_labels != -100
        n_lab = int(sel.sum())
        if n_lab == 0:
            dead_labels += 1
        else:
            loss_sum = F.cross_entropy(
                shift_logits[sel].float(), shift_labels[sel], reduction="sum"
            )
            nll_sum[domain] += loss_sum.item()
            tok_count[domain] += n_lab

        # --- 路由 / 专家：全部真实 token ---
        if is_moe:
            mask = batch["attention_mask"] == 1
            for li, layer in enumerate(model.model.layers):
                m = layer.mlp
                probs = torch.softmax(
                    m.last_router_big_logits[mask].float(), dim=-1
                )
                sci_w[li] += probs[:, 1].sum().double()
                expert_w[li] += m.last_little_topk_w[mask].float().sum(0).double()
            sci_tok += int(mask.sum())

    ppl = {
        d: math.exp(nll_sum[d] / tok_count[d]) for d in nll_sum if tok_count[d] > 0
    }
    result: dict[str, Any] = {
        "ppl": {d: round(ppl[d], 4) for d in DOMAINS if d in ppl},
        "ppl_tokens": {d: tok_count[d] for d in DOMAINS if d in tok_count},
        "samples_with_zero_loss": dead_labels,
        "sampled": len(indices),
    }
    if is_moe:
        # w 是 [n_layers, n_experts]：(层, 专家) 对的权重总和
        w = expert_w.cpu()
        total_pairs = n_layers * n_experts
        result["expert_health"] = {
            # 逐 (层, 专家) 对统计，而非按专家下标去重
            "experts_with_nonzero_weight": int((w > 0).sum()),
            "experts_never_called": int((w == 0).sum()),
            "total_experts": total_pairs,
            "per_layer_never_called": (w.sum(1) == 0).sum().item(),
            "layers_with_fully_dead_expert": int(((w == 0).any(dim=1)).sum()),
            "min_pair_weight_share": round(float((w / w.sum()).min()), 8),
        }
        sc = (sci_w.cpu() / max(sci_tok, 1)).tolist()
        result["routing_sci_share"] = [round(v, 4) for v in sc]
    return result


@torch.no_grad()
def routing_by_domain(
    model, dataset: DualContrastDataset, domains_map, device, batch_size, layers
) -> dict[str, dict[str, float]]:
    """逐 domain 采集每层的理科大核权重均值，用于展示路由分化。"""
    acc: dict[str, torch.Tensor] = {
        d: torch.zeros(len(layers), dtype=torch.float64, device=device)
        for d in DOMAINS
    }
    counts: dict[str, int] = {d: 0 for d in DOMAINS}
    layer_pos = {li: pos for pos, li in enumerate(layers)}

    for domain in DOMAINS:
        idx = domains_map.get(domain, [])
        for start in range(0, min(len(idx), 120), batch_size):
            chunk = idx[start : start + batch_size]
            items = [dataset[i] for i in chunk]
            ids = torch.stack([it["input_ids"] for it in items]).to(device)
            am = torch.stack([it["attention_mask"] for it in items]).to(device)
            model(input_ids=ids, attention_mask=am)
            mask = am == 1
            for li, layer in enumerate(model.model.layers):
                if li not in layer_pos:
                    continue
                probs = torch.softmax(
                    layer.mlp.last_router_big_logits[mask].float(), dim=-1
                )
                acc[domain][layer_pos[li]] += probs[:, 1].sum().double()
            counts[domain] += int(mask.sum())

    # 以层为外层键，便于与逐层视图对齐
    out: dict[str, dict[str, float]] = {}
    for pos, li in enumerate(layers):
        out[str(li)] = {
            d: round(float(acc[d][pos] / max(counts[d], 1)), 4) for d in DOMAINS
        }
    return out


# ----------------------------------------------------------------------
# 数值抖动 & 训练/推理一致性
# ----------------------------------------------------------------------
@torch.no_grad()
def jitter_check(model, dataset, device, runs: int, batch_size: int) -> dict:
    items = [dataset[i] for i in range(batch_size)]
    ids = torch.stack([it["input_ids"] for it in items]).to(device)
    am = torch.stack([it["attention_mask"] for it in items]).to(device)

    ref = model(input_ids=ids, attention_mask=am).logits.float()
    max_diff = 0.0
    bitwise = True
    for _ in range(runs - 1):
        out = model(input_ids=ids, attention_mask=am).logits.float()
        max_diff = max(max_diff, float((out - ref).abs().max()))
        bitwise &= bool(torch.equal(out, ref))
    return {
        "runs": runs,
        "max_abs_diff": max_diff,
        "bitwise_identical": bitwise,
    }


@torch.no_grad()
def impl_equivalence(
    train_model, infer_model, dataset, device, batch_size: int
) -> dict:
    """同一权重下，稠密训练路径与流式推理路径的输出差异。

    两条路径的 top-k 选择时机不同（逐位置 vs 末 token），prefill 下本就
    不等价；本检查量化差异幅度，供 README 显式说明，而非要求其相等。
    """
    items = [dataset[i] for i in range(batch_size)]
    ids = torch.stack([it["input_ids"] for it in items]).to(device)
    am = torch.stack([it["attention_mask"] for it in items]).to(device)

    a = train_model(input_ids=ids, attention_mask=am).logits.float()
    b = infer_model(input_ids=ids, attention_mask=am).logits.float()
    return {
        "batch": batch_size,
        "max_abs_diff": float((a - b).abs().max()),
        "mean_abs_diff": float((a - b).abs().mean()),
        "rel_max": float((a - b).abs().max() / a.abs().max().clamp_min(1e-9)),
    }


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def render(report: dict) -> str:
    L = ["=" * 68, "分域评估报告", "=" * 68]
    L.append(f"采样: {report['meta']['samples_per_domain']}/domain "
             f"(语料 in-distribution 切片，非泛化指标)")

    L.append("\n【分域困惑度】(越低越好)")
    L.append(f"  {'domain':<8}{'baseline':>12}{'双大核':>12}{'变化':>12}")
    b, d = report["baseline"], report["dual_train"]
    for dom in DOMAINS:
        if dom in b["ppl"] and dom in d["ppl"]:
            bv, dv = b["ppl"][dom], d["ppl"][dom]
            chg = (dv - bv) / bv * 100
            L.append(f"  {dom:<8}{bv:>12.2f}{dv:>12.2f}{chg:>11.1f}%")

    eh = d.get("expert_health")
    if eh:
        L.append("\n【专家健康度】")
        L.append(f"  获得非零权重的专家: {eh['experts_with_nonzero_weight']}"
                 f" / {eh['total_experts']}")
        L.append(f"  完全未调用的专家:   {eh['experts_never_called']} / {eh['total_experts']}")
        L.append(f"  存在饿死专家的层数: {eh['per_layer_never_called']}"
                 f" / 全部层有存活专家: "
                 f"{eh['total_experts'] - eh['experts_never_called'] > 0}")
        L.append(f"  单个(层,专家)对最小权重占比: {eh['min_pair_weight_share'] * 100:.6f}%")

    jit = report["jitter"]
    L.append("\n【数值抖动】")
    L.append(f"  {jit['runs']} 次相同前向  max|Δ| = {jit['max_abs_diff']:.3e}"
             f"  bitwise_identical = {jit['bitwise_identical']}")

    if "impl_equivalence" in report:
        e = report["impl_equivalence"]
        L.append("\n【训练路径 vs 推理路径】")
        L.append(f"  max|Δ| = {e['max_abs_diff']:.3e}   mean|Δ| = {e['mean_abs_diff']:.3e}"
                 f"   相对 = {e['rel_max']:.3e}")
        L.append("  注: 两者 top-k 选择时机不同（逐位置 vs 末 token），"
                 "prefill 下本就不等价")

    if "in_domain_routing" in report:
        L.append("\n【逐层路由分化 · 理科大核权重均值】")
        code = report["in_domain_routing"]
        L.append(f"  {'层':<6}{'Code':>10}{'Math':>10}{'Arts':>10}")
        for key in sorted(code, key=int):
            row = code[key]
            L.append(
                f"  {key:<6}"
                + "".join(
                    f"{row.get(d, 0.0):>10.3f}"
                    for d in ("Code", "Math", "Arts")
                )
            )
    L.append("=" * 68)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dual_contrast_data.jsonl")
    ap.add_argument("--weights", default="dual_big_resurrect_weights.pt")
    ap.add_argument("--out", default="reports/eval.json")
    ap.add_argument("--samples-per-domain", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--jitter-runs", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-infer", action="store_true",
                    help="跳过推理路径构建与一致性对比（更慢）")
    args = ap.parse_args()

    torch.manual_seed(0)
    dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    dataset = DualContrastDataset(args.data, tok)

    domains = read_domains(args.data)
    indices = pick_indices(domains, args.samples_per_domain)
    print(f"[eval] 采样 {len(indices)} 条，覆盖 "
          f"{{{', '.join(f'{k}:{len(v)}' for k, v in domains.items())}}}")

    report: dict[str, Any] = {
        "meta": {
            "base_model": MODEL_ID,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "samples_per_domain": args.samples_per_domain,
            "group_bounds": GROUP_BOUNDS,
            "note": "语料无 train/val 切分，本报告为 in-distribution 指标",
        }
    }

    t0 = time.time()
    base_model = build_baseline(args.device, dtype)
    report["baseline"] = analyze(base_model, dataset, indices, args.device, args.batch_size)
    del base_model
    torch.cuda.empty_cache()
    print(f"[eval] baseline 完成 ({time.time() - t0:.0f}s): {report['baseline']['ppl']}")

    t0 = time.time()
    train_model = build_dual_train_impl(args.weights, args.device, dtype)
    report["dual_train"] = analyze(train_model, dataset, indices, args.device, args.batch_size)
    report["jitter"] = jitter_check(train_model, dataset, args.device,
                                   args.jitter_runs, args.batch_size)
    report["in_domain_routing"] = routing_by_domain(
        train_model, dataset, domains, args.device, args.batch_size,
        layers=(0, 8, 16, 27),
    )
    print(f"[eval] 双大核(训练路径) 完成 ({time.time() - t0:.0f}s): "
          f"{report['dual_train']['ppl']}")

    if not args.skip_infer:
        t0 = time.time()
        infer_model = build_dual_infer_impl(args.weights, args.device, dtype)
        report["dual_infer"] = analyze(infer_model, dataset, indices,
                                       args.device, args.batch_size)
        report["impl_equivalence"] = impl_equivalence(
            train_model, infer_model, dataset, args.device, args.batch_size
        )
        print(f"[eval] 双大核(推理路径) 完成 ({time.time() - t0:.0f}s): "
              f"{report['dual_infer']['ppl']}")
        del infer_model
    del train_model
    torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 结果已写入 {out}")
    print()
    print(render(report))


if __name__ == "__main__":
    main()
