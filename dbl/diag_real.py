"""在真实训练里定位: 为什么 router 学不动。

前面的微缩实验已证明 aux 损失本身可学、梯度充足。
这里在真实 Qwen3-0.6B + 真实语料上逐步观测：
  * 训练前 20 步内 aux loss 的变化
  * router_big 权重的实际变化量
  * 标签与 logits 的相关性
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dbl.config import Config
from dbl.data import DualContrastDataset
from dbl.runtime import format_report, resolve_device
from dbl.train import aux_losses, build_model, set_seed

#: 诊断脚本统一走设备解析（auto / cpu / cuda:N），不再硬编码卡号。
#: 刻意保持为 "auto" 字面量而非导入期解析结果——否则本模块在没有
#: 显卡的机器上连 import 都会失败。
DEV = "auto"


def main():
    global DEV
    DEV = resolve_device(DEV, allow_cpu=False)
    print(format_report())
    cfg = Config(micro_batch=2, grad_accum_steps=2, num_workers=0, seed=0)
    set_seed(cfg.seed)
    model, mods, tok = build_model(cfg)
    ds = DualContrastDataset(cfg.data_path, tok, cfg)
    loader = DataLoader(ds, batch_size=cfg.micro_batch, shuffle=True,
                        num_workers=0, drop_last=True)

    print("=" * 70)
    print("真实训练前 12 步的辅助损失演化")
    print("=" * 70)
    print(f"  {'step':>5}{'big_aux':>10}{'little_aux':>12}{'lm':>10}"
          f"{'router_acc':>12}{'|W|变化':>12}")

    w0 = mods[0].router_big.weight.detach().clone()
    opt = torch.optim.AdamW(mods[0].parameter_groups())

    for i, batch in enumerate(loader):
        if i >= 12:
            break
        batch = {k: v.to(DEV) for k, v in batch.items()}
        for m in mods:
            m.collect_stats = True
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        lm = out.loss
        mask = batch["attention_mask"] == 1
        big_l, little_l, bal_l = aux_losses(mods, [m.last_stats for m in mods],
                                            batch["big_target"],
                                            batch["group_target"], mask,
                                            cfg.groups)
        raw = (lm + cfg.router_aux_weight * big_l
               + cfg.router_aux_weight * little_l
               + cfg.load_balance_weight * bal_l)
        (raw / 2).backward()
        if (i + 1) % 2 == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

        # 当前 batch 上的大核路由准确率
        with torch.no_grad():
            p = torch.softmax(mods[0].last_stats.big_logits[mask].float(), -1)
            gt = batch["big_target"].unsqueeze(1).expand(
                -1, batch["input_ids"].shape[1])[mask]
            acc = (p.argmax(-1) == gt).float().mean().item()
        dw = float((mods[0].router_big.weight - w0).abs().max())
        print(f"  {i + 1:>5}{float(big_l):>10.4f}{float(little_l):>12.4f}"
              f"{float(lm):>10.4f}{acc:>12.3f}{dw:>12.2e}")
        for m in mods:
            m.collect_stats = False

    print()
    print("参考：二分类随机 CE=0.6931  三分类随机 CE=1.0986")
    print()
    print("标签分布检查：")
    import collections
    print("  big_target:", dict(collections.Counter(
        r["big_target"] for r in ds.rows)))
    print("  group_target:", dict(collections.Counter(
        r["little_group"] for r in ds.rows)))


if __name__ == "__main__":
    main()
