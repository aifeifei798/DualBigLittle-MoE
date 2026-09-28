#!/usr/bin/env python
"""分域评估与可复现性体检。

单次前向遍历同时采集四类指标，用于替代 README 中手写的实测表格：

1. 分域困惑度（PPL）    —— Code / Math / Arts，与原生 Qwen3-0.6B baseline 对比
2. 专家健康度          —— 获得非零权重的 (层,专家) 对数、完全未被调用数
3. 路由分化            —— 逐层 big_sci（理科大核）权重均值
4. 数值抖动            —— 同一输入重复前向的输出差异

并额外比对训练路径（逐位置稠密 top-k）与推理路径（末 token top-k +
pinned 流式搬运）的差异 —— 两者在 decode 阶段等价，prefill 下本就不同。

用法::

    python eval_ppl.py --out reports/eval.json
    python eval_ppl.py --split val --out reports/eval_val.json

注意：若语料未经阶段 C 的 train/val 切分，默认评估的是同分布切片，
属于 in-distribution 指标，不等同于泛化性能。
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

from dbl.checkpoint import apply_checkpoint, load_checkpoint
from dbl.config import Config
from dbl.data import DualContrastDataset, load_jsonl
from dbl.moe import TrainMoE, inject_moe
from dbl.runtime import device_report, fail_cli, format_report, resolve_device

DOMAINS = ("Code", "Math", "Arts")


# ----------------------------------------------------------------------
# 采样
# ----------------------------------------------------------------------
def pick_indices(domains: dict[str, list[int]], per_domain: int) -> list[int]:
    """每个 domain 内等距取样，确定性且覆盖整个分布。"""
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


def index_by_domain(rows: list[dict]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r.get("domain", "Arts")].append(i)
    return dict(buckets)


# ----------------------------------------------------------------------
# 模型构建
# ----------------------------------------------------------------------
def _load_base(cfg: Config, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.torch_dtype, device_map=device
    )
    model.eval()
    return model


def build_baseline(cfg: Config, device: str):
    return _load_base(cfg, device)


def build_dual(cfg: Config, weights: str, device: str, mode: str):
    """构造双大核模型并装载权重。

    结构超参优先取 checkpoint 自带的 config —— 这正是本次重构消除
    "训练/推理各维护一份常量" 的落点。
    """
    payload = load_checkpoint(weights, map_location="cpu")
    ck_cfg = payload.get("cfg")
    if ck_cfg is not None:
        cfg = ck_cfg.replace(device=device)
    model = _load_base(cfg, device)
    if mode == "train":
        for p in model.parameters():
            p.requires_grad = False
    mods = inject_moe(model, cfg, mode=mode)
    apply_checkpoint(mods, payload)
    if mode == "infer":
        for m in mods:
            m.load_expert_pool(m.lora_A.data, m.lora_B.data)
    model.eval()
    return model, cfg, payload


# ----------------------------------------------------------------------
# 单次前向采集
# ----------------------------------------------------------------------
class _StatsHook:
    """用 forward hook 收集每层 RouterStats。"""

    def __init__(self, modules):
        self.modules = modules
        self.collected: list = []
        for m in modules:
            m.collect_stats = True
        self._handles = [
            m.register_forward_hook(self._mk()) for m in modules
        ]

    def _mk(self):
        def hook(_m, _inp, out):
            # 只读 side channel，不改返回值（见 TrainMoE.collect_stats 注释）
            self.collected.append(getattr(_m, "last_stats", None))
        return hook

    def clear(self):
        self.collected = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for m in self.modules:
            m.collect_stats = False


@torch.no_grad()
def analyze(
    model,
    dataset: DualContrastDataset,
    indices: list[int],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """一次遍历收集 PPL / 专家健康度 / 路由分化。

    PPL 走因果 LM 标准移位且只统计 ``label != -100`` 的 response 位置，
    与训练目标（prompt 掩码）一致。
    路由与专家统计走 ``attention_mask``，即全部真实 token，
    与训练时辅助损失的口径相同。
    """
    is_moe = isinstance(getattr(model.model.layers[0], "mlp", None), TrainMoE)
    layers = list(model.model.layers)
    n_layers, n_experts = len(layers), 0

    nll_sum: dict[str, float] = defaultdict(float)
    tok_count: dict[str, int] = defaultdict(int)
    dead_labels = 0
    expert_w = sci_w = None
    sci_tok = 0

    hook = _StatsHook([lyr.mlp for lyr in layers]) if is_moe else None
    if is_moe:
        n_experts = layers[0].mlp.num_experts
        expert_w = torch.zeros(n_layers, n_experts, dtype=torch.float64, device=device)
        sci_w = torch.zeros(n_layers, dtype=torch.float64, device=device)

    try:
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            items = [dataset[i] for i in chunk]
            batch = {
                k: torch.stack([it[k] for it in items]).to(device, non_blocking=True)
                for k in ("input_ids", "attention_mask", "labels")
            }
            domain = dataset.rows[chunk[0]]["domain"]

            if hook:
                hook.clear()
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            logits = out.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            sel = shift_labels != -100
            n_lab = int(sel.sum())
            if n_lab == 0:
                dead_labels += 1
            else:
                nll_sum[domain] += F.cross_entropy(
                    shift_logits[sel].float(), shift_labels[sel], reduction="sum"
                ).item()
                tok_count[domain] += n_lab

            if hook:
                mask = batch["attention_mask"] == 1
                for li, st in enumerate(hook.collected):
                    if st is None:
                        continue
                    probs = torch.softmax(st.big_logits[mask].float(), -1)
                    sci_w[li] += probs[:, 1].sum().double()
                    expert_w[li] += st.topk_weights[mask].float().sum(0).double()
                sci_tok += int(mask.sum())
    finally:
        if hook:
            hook.remove()

    ppl = {d: math.exp(nll_sum[d] / tok_count[d]) for d in nll_sum if tok_count[d]}
    result: dict[str, Any] = {
        "ppl": {d: round(ppl[d], 4) for d in DOMAINS if d in ppl},
        "ppl_tokens": {d: tok_count[d] for d in DOMAINS if d in tok_count},
        "samples_with_zero_loss": dead_labels,
        "sampled": len(indices),
    }
    if is_moe:
        w = expert_w.cpu()
        result["expert_health"] = {
            # 逐 (层, 专家) 对统计，而非按专家下标去重
            "experts_with_nonzero_weight": int((w > 0).sum()),
            "experts_never_called": int((w == 0).sum()),
            "total_experts": n_layers * n_experts,
            "per_layer_never_called": (w.sum(1) == 0).sum().item(),
            "min_pair_weight_share": round(float((w / w.sum()).min()), 10),
        }
        result["routing_sci_share"] = [
            round(float(v), 4) for v in (sci_w.cpu() / max(sci_tok, 1))
        ]
    return result


# ----------------------------------------------------------------------
# 抖动 / 路径一致性 / 路由分化
# ----------------------------------------------------------------------
@torch.no_grad()
def jitter_check(model, dataset, device, runs, batch_size):
    items = [dataset[i] for i in range(batch_size)]
    ids = torch.stack([it["input_ids"] for it in items]).to(device)
    am = torch.stack([it["attention_mask"] for it in items]).to(device)
    ref = model(input_ids=ids, attention_mask=am).logits.float()
    max_diff, bitwise = 0.0, True
    for _ in range(runs - 1):
        o = model(input_ids=ids, attention_mask=am).logits.float()
        max_diff = max(max_diff, float((o - ref).abs().max()))
        bitwise &= bool(torch.equal(o, ref))
    return {"runs": runs, "max_abs_diff": max_diff, "bitwise_identical": bitwise}


@torch.no_grad()
def impl_equivalence(train_model, infer_model, dataset, device, batch_size):
    """量化 prefill 下两条路径的差异（decode 阶段应等价，见 tests）。"""
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


@torch.no_grad()
def routing_by_domain(model, dataset, domains, device, batch_size, layers_wanted):
    """逐 domain 采集每层的理科大核权重均值。"""
    is_moe = isinstance(getattr(model.model.layers[0], "mlp", None), TrainMoE)
    if not is_moe:
        return {}
    pos = {li: p for p, li in enumerate(layers_wanted)}
    acc = {d: torch.zeros(len(layers_wanted), dtype=torch.float64, device=device)
           for d in DOMAINS}
    counts = {d: 0 for d in DOMAINS}
    hook = _StatsHook([lyr.mlp for lyr in model.model.layers])
    try:
        for domain in DOMAINS:
            idx = domains.get(domain, [])[:120]
            for start in range(0, len(idx), batch_size):
                chunk = idx[start : start + batch_size]
                items = [dataset[i] for i in chunk]
                ids = torch.stack([it["input_ids"] for it in items]).to(device)
                am = torch.stack([it["attention_mask"] for it in items]).to(device)
                hook.clear()
                model(input_ids=ids, attention_mask=am)
                mask = am == 1
                for li, st in enumerate(hook.collected):
                    if st is None or li not in pos:
                        continue
                    probs = torch.softmax(st.big_logits[mask].float(), -1)
                    acc[domain][pos[li]] += probs[:, 1].sum().double()
                counts[domain] += int(mask.sum())
    finally:
        hook.remove()
    return {
        str(li): {d: round(float(acc[d][p] / max(counts[d], 1)), 4)
                  for d in DOMAINS}
        for p, li in enumerate(layers_wanted)
    }


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def render(rep: dict) -> str:
    L = ["=" * 68, "分域评估报告", "=" * 68]
    L.append(f"采样: {rep['meta']['samples_per_domain']}/domain · "
             f"split={rep['meta']['split']}")
    L.append(f"语料: {rep['meta']['data_path']}")
    if rep["meta"].get("holdout"):
        L.append("※ 使用 train/val 切分出的验证集")
    else:
        L.append("※ 未切分，指标为 in-distribution，不能等同泛化性能")

    b, d = rep["baseline"], rep["dual_train"]
    L.append("\n【分域困惑度】(越低越好)")
    L.append(f"  {'domain':<8}{'baseline':>12}{'双大核':>12}{'变化':>12}")
    for dom in DOMAINS:
        if dom in b["ppl"] and dom in d["ppl"]:
            bv, dv = b["ppl"][dom], d["ppl"][dom]
            L.append(f"  {dom:<8}{bv:>12.2f}{dv:>12.2f}"
                     f"{(dv - bv) / bv * 100:>11.1f}%")
    if "dual_infer" in rep:
        di = rep["dual_infer"]
        L.append(f"  {'(推理路径)':<8}{'':>12}"
                 f"{di['ppl'].get('Code', 0):>12.2f}")
        L.append("  注: 推理路径 prefill 用末 token 选专家，与训练路径略有差异")

    eh = d.get("expert_health")
    if eh:
        L.append("\n【专家健康度】")
        L.append(f"  获得非零权重的 (层,专家) 对: "
                 f"{eh['experts_with_nonzero_weight']} / {eh['total_experts']}")
        L.append(f"  完全未调用的:               "
                 f"{eh['experts_never_called']} / {eh['total_experts']}")
        L.append(f"  存在饿死专家的层数:         {eh['per_layer_never_called']}")
        L.append(f"  单对最小权重占比:           "
                 f"{eh['min_pair_weight_share'] * 100:.8f}%")

    j = rep["jitter"]
    L.append("\n【数值抖动】")
    L.append(f"  {j['runs']} 次相同前向  max|Δ| = {j['max_abs_diff']:.3e}"
             f"   逐位一致 = {j['bitwise_identical']}")

    if "impl_equivalence" in rep:
        e = rep["impl_equivalence"]
        L.append("\n【训练路径 vs 推理路径 (prefill)】")
        L.append(f"  max|Δ| = {e['max_abs_diff']:.3e}  "
                 f"mean|Δ| = {e['mean_abs_diff']:.3e}  相对 = {e['rel_max']:.3e}")
        L.append("  注: 两者 top-k 选择时机不同（逐位置 vs 末 token），"
                 "prefill 下本就不等价；decode 阶段等价由 tests 保证")

    if rep.get("in_domain_routing"):
        L.append("\n【逐层路由分化 · 理科大核权重均值】")
        L.append(f"  {'层':<6}{'Code':>10}{'Math':>10}{'Arts':>10}")
        for key in sorted(rep["in_domain_routing"], key=int):
            row = rep["in_domain_routing"][key]
            L.append(f"  {key:<6}" + "".join(
                f"{row.get(d, 0.0):>10.3f}" for d in ("Code", "Math", "Arts")
            ))
    L.append("=" * 68)
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dual_contrast_data.jsonl")
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--weights", default="dual_big_resurrect_weights.pt")
    ap.add_argument("--out", default="reports/eval.json")
    ap.add_argument("--samples-per-domain", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--jitter-runs", type=int, default=20)
    ap.add_argument("--device", default="auto",
                    help="auto / cpu / cuda / cuda:N，默认自动探测")
    ap.add_argument("--skip-infer", action="store_true")
    args = ap.parse_args()

    try:
        args.device = resolve_device(args.device, allow_cpu=False)
    except (RuntimeError, ValueError) as exc:
        fail_cli(exc, "eval_ppl.py")
        return
    print(format_report())
    print(f"[eval] 设备 {args.device}")

    torch.manual_seed(0)
    base_cfg = Config(data_path=args.data)

    tok = AutoTokenizer.from_pretrained(base_cfg.model_id)
    tok.pad_token = tok.eos_token
    dataset = DualContrastDataset(args.data, tok, base_cfg)
    rows = dataset.rows
    if args.split != "all":
        rows = load_jsonl(f"dual_contrast_{args.split}.jsonl")
        dataset = DualContrastDataset(
            f"dual_contrast_{args.split}.jsonl", tok, base_cfg
        )
    domains = index_by_domain(rows)
    indices = pick_indices(domains, args.samples_per_domain)
    print(f"[eval] 采样 {len(indices)} 条 | "
          f"{ {k: len(v) for k, v in domains.items()} }")

    dev = device_report()
    report: dict[str, Any] = {
        "meta": {
            "base_model": base_cfg.model_id,
            "torch": torch.__version__,
            "device": args.device,
            "gpu": dev.get("gpu", "cpu"),
            "compute_capability": dev.get("compute_capability"),
            "torch_cuda_build": dev.get("torch_cuda_build"),
            "samples_per_domain": args.samples_per_domain,
            "split": args.split,
            "data_path": args.data if args.split == "all"
            else f"dual_contrast_{args.split}.jsonl",
            "holdout": args.split == "val",
            "group_bounds": list(base_cfg.groups.bounds),
        }
    }

    t0 = time.time()
    base_model = build_baseline(base_cfg, args.device)
    report["baseline"] = analyze(base_model, dataset, indices, args.device,
                                 args.batch_size)
    del base_model
    torch.cuda.empty_cache()
    print(f"[eval] baseline ({time.time() - t0:.0f}s): {report['baseline']['ppl']}")

    t0 = time.time()
    tr_model, cfg, payload = build_dual(base_cfg, args.weights, args.device, "train")
    report["dual_train"] = analyze(tr_model, dataset, indices, args.device,
                                  args.batch_size)
    report["jitter"] = jitter_check(tr_model, dataset, args.device,
                                   args.jitter_runs, args.batch_size)
    report["in_domain_routing"] = routing_by_domain(
        tr_model, dataset, domains, args.device, args.batch_size, (0, 8, 16, 27)
    )
    report["meta"]["checkpoint"] = {
        "format": payload.get("kind"),
        "delta_dtype": payload.get("meta", {}).get("delta_dtype"),
        "quant_rel_error": payload.get("meta", {}).get("quant_rel_error"),
    }
    print(f"[eval] 双大核/训练路径 ({time.time() - t0:.0f}s): "
          f"{report['dual_train']['ppl']}")

    if not args.skip_infer:
        t0 = time.time()
        inf_model, _, _ = build_dual(base_cfg, args.weights, args.device, "infer")
        report["dual_infer"] = analyze(inf_model, dataset, indices, args.device,
                                      args.batch_size)
        report["impl_equivalence"] = impl_equivalence(
            tr_model, inf_model, dataset, args.device, args.batch_size
        )
        print(f"[eval] 双大核/推理路径 ({time.time() - t0:.0f}s): "
              f"{report['dual_infer']['ppl']}")
        del inf_model
    del tr_model
    torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[eval] 已写入 {out}\n")
    print(render(report))


if __name__ == "__main__":
    main()
