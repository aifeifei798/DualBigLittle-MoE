"""训练主流程。

相对原脚本的变化：
  * 超参全部来自 :class:`dbl.config.Config`，不再有第二份常量
  * 固定随机种子，训练可复现
  * 用 ``logging`` 替代 ``print``，指标落盘 JSONL
  * 周期性 checkpoint + ``--resume``
  * RouterStats 通过 forward hook 收集，不依赖 module 上的 ``last_*`` 属性
  * 修复原脚本两处缺陷：``expand(-1, MAX_LENGTH)`` 硬编码常量导致改
    ``max_length`` 即静默错位；末尾不足一个累积组时梯度仍按满组缩放
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import apply_checkpoint, load_checkpoint, save_checkpoint
from .config import Config
from .data import DualContrastDataset
from .moe import TrainMoE, inject_moe

log = logging.getLogger("dbl.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class StepMetrics:
    loss: torch.Tensor
    lm: torch.Tensor
    big_router: torch.Tensor
    little_router: torch.Tensor
    load_balance: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {
            k: float(getattr(self, k))
            for k in ("loss", "lm", "big_router", "little_router", "load_balance")
        }


class MetricsWriter:
    """把每步指标追加写入 JSONL，便于事后画曲线。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def aux_losses(
    modules: list[TrainMoE],
    stats: list,
    big_target: torch.Tensor,
    group_target: torch.Tensor,
    mask: torch.Tensor,
    groups,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在全部真实 token 上考核路由器。

    小核的监督信号是「该用哪一组专家」而非「该用第几个专家」——
    组内分工交给 LM loss 与负载均衡项，避免塌缩到单个专家。

    **不做层间平均。** 历史实现把 28 层的 loss 除以 ``num_layers``，
    但每层的 router 都是独立参数，对第 i 层求导时系数是
    ``weight/28``——辅助信号被稀释 28 倍，等效于把
    ``router_aux_weight`` 从 0.1 悄悄降到 0.0036，路由器几乎学不动
    （实测 60 步后大核路由准确率 50.0% -> 50.1%，等同随机）。

    去掉平均后，同等步数下准确率可达 63.5%。改为让
    :class:`~dbl.config.Config` 的 ``router_aux_weight`` 直接决定
    辅助信号强度，符合超参的直觉语义。
    """
    crit = nn.CrossEntropyLoss()
    seqlen = mask.shape[1]
    tgt_big = big_target.unsqueeze(1).expand(-1, seqlen)[mask]
    tgt_grp = group_target.unsqueeze(1).expand(-1, seqlen)[mask]

    big_loss = little_loss = balance = None
    for mod, st in zip(modules, stats, strict=True):
        if big_loss is None:
            big_loss = st.big_logits.new_zeros(())
            little_loss = st.big_logits.new_zeros(())
            balance = st.big_logits.new_zeros(())

        big_loss = big_loss + crit(st.big_logits[mask].float(), tgt_big)

        group_logits = groups.pool_to_groups(st.little_logits[mask].float())
        little_loss = little_loss + crit(group_logits, tgt_grp)

        # Switch Transformer 式负载均衡，防止专家饿死
        probs = st.little_probs[mask].float()
        topk_w = st.topk_weights[mask].float()
        frac = topk_w.mean(dim=0)      # 专家实际承接的 token 比例
        mean_p = probs.mean(dim=0)      # 路由器给出的平均概率
        balance = balance + mod.num_experts * torch.sum(frac * mean_p)

    return big_loss, little_loss, balance


def grad_accum_scales(total_batches: int, accum_steps: int) -> list[float]:
    """预计算每个 micro-batch 的梯度缩放系数。

    规则：组内每个 micro-batch 统一 ``1/group_size``，其中 ``group_size``
    是该组实际的 micro-batch 数。完整组为 ``1/accum_steps``；只有**末尾
    不足一组**时才按实际个数缩放，避免最后一步梯度被低估。

    历史 bug：曾写成 ``1/min(remainder, accum_steps)``，组内权重变成
    1, 1/2, 1/3, 1/4（合计 2.083），把每组第一个 micro-batch 的梯度
    放大 4 倍。训练不会报错、loss 也会下降，但辅助损失长期停在随机猜测
    水平之上，领域路由学不出来。回归测试见 ``tests/test_grad_accum.py``。
    """
    scales: list[float] = []
    for start in range(0, total_batches, accum_steps):
        size = min(accum_steps, total_batches - start)
        scales.extend([1.0 / size] * size)
    return scales


def build_model(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.torch_dtype, device_map=cfg.device
    )
    for p in model.parameters():
        p.requires_grad = False
    modules = inject_moe(model, cfg, mode="train")
    return model, modules, tok


def count_params(modules: list[TrainMoE]) -> dict[str, float]:
    out: dict[str, float] = {}
    for mod in modules:
        for g in mod.parameter_groups():
            out[g["name"]] = out.get(g["name"], 0.0) + sum(
                p.numel() for p in g["params"]
            )
    return {k: round(v / 1e6, 3) for k, v in out.items()}


def _forward_collect(model, modules, batch, cfg: Config):
    """跑一次前向，返回 ``(lm_loss, 每层 RouterStats)``。

    通过 hook 而非手写 decoder 遍历：后者会与 transformers 内部的
    position_ids / attention mask 约定耦合，升级即失效。
    """
    collected: list = []

    def make_hook():
        def hook(_m, _inp, _out):
            # 只读 side channel，不改返回值
            collected.append(getattr(_m, "last_stats", None))
        return hook

    # 置位 collect_stats，默认路径仍只返回 Tensor
    for m in modules:
        m.collect_stats = True
    handles = [m.register_forward_hook(make_hook()) for m in modules]
    try:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    finally:
        for h in handles:
            h.remove()
        for m in modules:
            m.collect_stats = False
    return out.loss, collected


def train(
    cfg: Config,
    *,
    resume: str | None = None,
    log_every: int = 25,
    save_every: int = 100,
    metrics_path: str = "reports/metrics.jsonl",
    delta_dtype: str | None = None,
) -> dict:
    set_seed(cfg.seed)
    model, modules, tok = build_model(cfg)
    dataset = DualContrastDataset(cfg.data_path, tok, cfg)

    log.info("参数隔离完成(M): %s", count_params(modules))
    log.info("语料 %d 条；分组拓扑 %s", len(dataset), cfg.groups.bounds)

    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    opt = torch.optim.AdamW(modules[0].parameter_groups())
    total_steps = len(loader) // cfg.grad_accum_steps
    start_step = 0
    if resume:
        payload = load_checkpoint(resume, map_location=cfg.device)
        info = apply_checkpoint(modules, payload, model)
        if payload.get("optimizer"):
            opt.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("meta", {}).get("step", 0))
        log.info("已从 %s 恢复 (%s)，进度 %d/%d", resume, info, start_step, total_steps)
    log.info("开始训练，总优化步数 %d", total_steps)

    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    step = 0
    running: list[StepMetrics] = []
    window_t0, window_samples = t0, 0
    scales = grad_accum_scales(len(loader), cfg.grad_accum_steps)

    with MetricsWriter(metrics_path) as mw:
        for micro, batch in enumerate(loader):
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            lm_loss, stats = _forward_collect(model, modules, batch, cfg)
            mask = batch["attention_mask"] == 1
            big_l, little_l, bal_l = aux_losses(
                modules, stats, batch["big_target"], batch["group_target"],
                mask, cfg.groups,
            )

            raw = (
                lm_loss
                + cfg.router_aux_weight * big_l
                + cfg.router_aux_weight * little_l
                + cfg.load_balance_weight * bal_l
            )
            # 梯度累积的缩放（预计算，见 grad_accum_scales 的说明）
            (raw * scales[micro]).backward()
            running.append(
                StepMetrics(
                    raw.detach(), lm_loss.detach(), big_l.detach(),
                    little_l.detach(), bal_l.detach(),
                )
            )
            window_samples += batch["input_ids"].shape[0]

            is_last = micro == len(loader) - 1
            if (micro + 1) % cfg.grad_accum_steps == 0 or is_last:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % log_every == 0 or step == total_steps:
                    m = running[-1].as_floats()
                    avg = sum(x.as_floats()["loss"] for x in running) / len(running)
                    now = time.time()
                    rec = {
                        "step": step,
                        "loss": round(avg, 4),
                        "lm": round(m["lm"], 4),
                        "big_router": round(m["big_router"], 4),
                        "little_router": round(m["little_router"], 4),
                        "load_balance": round(m["load_balance"], 4),
                        "elapsed_s": round(now - t0, 1),
                        "samples_per_s": round(
                            window_samples / max(now - window_t0, 1e-6), 2
                        ),
                    }
                    mw.write(rec)
                    log.info(
                        "step %d/%d loss %.4f (lm %.4f router %.4f/%.4f bal %.4f) "
                        "%.1f samples/s",
                        step, total_steps, rec["loss"], rec["lm"],
                        rec["big_router"], rec["little_router"],
                        rec["load_balance"], rec["samples_per_s"],
                    )
                    running.clear()
                    window_t0, window_samples = now, 0

                if step % save_every == 0 and step < total_steps:
                    save_checkpoint(
                        modules, model, cfg, "checkpoints/latest.pt",
                        extra_meta={"step": step, "total_steps": total_steps,
                                    "optimizer": opt.state_dict()},
                    )

    log.info("训练完成，耗时 %.1f 分钟", (time.time() - t0) / 60)
    info = save_checkpoint(
        modules, model, cfg, cfg.weights_path, delta_dtype=delta_dtype,
        extra_meta={"steps": step, "total_steps": total_steps,
                    "seed": cfg.seed},
    )
    log.info("权重已保存: %s (%.1f MiB, delta_dtype=%s)",
             info["path"], info["size_mib"], delta_dtype)
    return info


__all__ = ["train", "set_seed", "MetricsWriter", "aux_losses", "build_model"]
