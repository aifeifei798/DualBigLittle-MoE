"""假设检验: 路由器学不动是因为 aux 被 1/num_layers 稀释 + 学习率过低。

对照实验（其余全部相同，只改一个变量）:
  A. 现状        : aux/28, lr_router=3e-4
  B. 不除层数     : aux 原样, lr_router=3e-4
  C. 提高路由 lr  : aux/28, lr_router=3e-3
  D. 两者都改
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dbl.config import Config
from dbl.data import DualContrastDataset
from dbl.train import aux_losses, build_model, grad_accum_scales, set_seed

DEV = "cuda:0"
STEPS = 60


@torch.no_grad()
def eval_router(model, mods, ds, cfg, n_batches=24):
    """在**固定**验证批上评估大核路由，避免单 batch 噪声掩盖趋势。"""
    loader = DataLoader(ds, batch_size=cfg.micro_batch, shuffle=False,
                        num_workers=0, drop_last=True)
    crit = nn.CrossEntropyLoss()
    tot, ntok, corr = 0.0, 0, 0
    was_training = model.training
    model.eval()
    for m in mods:
        m.collect_stats = True
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(DEV) for k, v in batch.items()}
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        mask = batch["attention_mask"] == 1
        for m in mods:
            lg = m.last_stats.big_logits[mask].float()
            gt = batch["big_target"].unsqueeze(1).expand(
                -1, batch["input_ids"].shape[1])[mask]
            tot += float(crit(lg, gt)) * lg.shape[0]
            corr += int((lg.argmax(-1) == gt).sum())
            ntok += lg.shape[0]
    for m in mods:
        m.collect_stats = False
    if was_training:
        model.train()
    return tot / max(ntok, 1), corr / max(ntok, 1)


def run(tag, aux_w, lr_router, steps=STEPS, bal_w=0.01):
    cfg = Config(router_aux_weight=aux_w, load_balance_weight=bal_w,
                 lr_router=lr_router)
    set_seed(cfg.seed)
    model, mods, tok = build_model(cfg)
    ds = DualContrastDataset(cfg.data_path, tok, cfg)
    loader = DataLoader(ds, batch_size=cfg.micro_batch, shuffle=True,
                        num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    scales = grad_accum_scales(len(loader), cfg.grad_accum_steps)
    opt = torch.optim.AdamW(mods[0].parameter_groups())
    model.train()
    opt.zero_grad(set_to_none=True)

    l0, a0 = eval_router(model, mods, ds, cfg)
    for i, batch in enumerate(loader):
        if i >= steps * cfg.grad_accum_steps:
            break
        batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
        for m in mods:
            m.collect_stats = True
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        mask = batch["attention_mask"] == 1
        big_l, little_l, bal_l = aux_losses(mods, [m.last_stats for m in mods],
                                            batch["big_target"],
                                            batch["group_target"], mask,
                                            cfg.groups)
        raw = (out.loss + cfg.router_aux_weight * (big_l + little_l)
               + cfg.load_balance_weight * bal_l)
        (raw * scales[i]).backward()
        for m in mods:
            m.collect_stats = False
        if (i + 1) % cfg.grad_accum_steps == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

    l1, a1 = eval_router(model, mods, ds, cfg)
    print(f"  aux_w={aux_w:<7} lr_router={lr_router:<8} "
          f"CE {l0:.4f} -> {l1:.4f}   acc {a0:.1%} -> {a1:.1%}")
    del model
    torch.cuda.empty_cache()
    return l1, a1


def main():
    print("=" * 72)
    print(f"辅助权重扫描：{STEPS} 优化步，固定验证批评估")
    print("参考：二分类随机 CE=0.6931，acc=50%（越低越好）")
    print("旧行为（/28，aux_w=0.1）折算后等效于 aux_w≈0.0036")
    print("=" * 72)
    for aux_w in (0.0036, 0.01, 0.03, 0.1):
        run("", aux_w, 3e-4)
    print()
    for lr in (1e-3, 3e-3):
        run("", 0.03, lr)


if __name__ == "__main__":
    main()
