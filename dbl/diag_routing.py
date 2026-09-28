"""诊断: 路由分化变弱是语料导致还是训练导致。

隔离实验: 同一份 checkpoint 分别在旧语料与新语料上测路由权重。
若旧 ckpt 在新语料上路由依然强 -> 语料问题;
若旧 ckpt 在新语料上也弱 -> 语料问题(已训练好的权重对语料敏感);
若新 ckpt 在旧语料上强 -> 训练问题。
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dbl.checkpoint import apply_checkpoint, load_checkpoint
from dbl.config import Config
from dbl.data import DualContrastDataset
from dbl.moe import inject_moe

DOMAINS = ("Code", "Math", "Arts")
LAYERS = (0, 8, 16, 27)


@torch.no_grad()
def routing(weights, data_path, device, n_per_domain, batch_size, limit_chars=0):
    payload = load_checkpoint(weights, map_location="cpu")
    cfg = payload.get("cfg") or Config()
    cfg = cfg.replace(device=device)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.pad_token = tok.eos_token
    ds = DualContrastDataset(data_path, tok, cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.torch_dtype, device_map=device
    )
    for p in model.parameters():
        p.requires_grad = False
    mods = inject_moe(model, cfg, mode="train")
    apply_checkpoint(mods, payload)
    model.eval()
    for m in mods:
        m.collect_stats = True

    rows = ds.rows
    by_dom: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_dom.setdefault(r.get("domain", "Arts"), []).append(i)

    n_layers = len(mods)
    acc = {d: torch.zeros(n_layers, dtype=torch.float64, device=device)
           for d in DOMAINS}
    cnt = {d: 0 for d in DOMAINS}
    sel_big = {d: torch.zeros(n_layers, dtype=torch.float64, device=device)
               for d in DOMAINS}   # aux 监督下的 top-1 命中率

    for dom in DOMAINS:
        idx = by_dom.get(dom, [])[:n_per_domain]
        for s in range(0, len(idx), batch_size):
            chunk = idx[s : s + batch_size]
            items = [ds[i] for i in chunk]
            ids = torch.stack([it["input_ids"] for it in items]).to(device)
            am = torch.stack([it["attention_mask"] for it in items]).to(device)
            gt = torch.stack([it["big_target"] for it in items]).to(device)
            model(input_ids=ids, attention_mask=am)
            mask = am == 1
            for li, m in enumerate(mods):
                p = torch.softmax(m.last_stats.big_logits[mask].float(), -1)
                acc[dom][li] += p[:, 1].sum().double()
                sel_big[dom][li] += (
                    (p.argmax(-1) == gt.unsqueeze(1).expand(-1, ids.shape[1])[mask])
                    .sum().double()
                )
            cnt[dom] += int(mask.sum())

    out = {}
    for dom in DOMAINS:
        n = max(cnt[dom], 1)
        out[dom] = {
            "sci_weight": {str(li): round(float(acc[dom][li] / n), 4)
                           for li in LAYERS if li < n_layers},
            "router_acc": {str(li): round(float(sel_big[dom][li] / n), 4)
                           for li in LAYERS if li < n_layers},
        }
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/routing_diag.json")
    ap.add_argument("--data", nargs="+", required=True,
                    help="多个 标签=路径 组合")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="多个 标签=路径 组合")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    report = {}
    for wspec in args.weights:
        wlabel, wpath = wspec.split("=", 1)
        for dspec in args.data:
            dlabel, dpath = dspec.split("=", 1)
            if not (Path_exists(dpath)):
                print(f"  跳过缺失语料 {dpath}")
                continue
            r = routing(wpath, dpath, args.device, args.n, args.batch_size)
            report[f"{wlabel}|{dlabel}"] = r
            print(f"\n=== {wlabel}  @  {dlabel} ===")
            print(f"  {'层':<5}{'Code sci':>10}{'Math sci':>10}{'Arts sci':>10}"
                  f"{'C 差':>8}{'A 差':>8}")
            for lbl in LAYERS:
                k = str(lbl)
                c = r["Code"]["sci_weight"].get(k)
                m = r["Math"]["sci_weight"].get(k)
                a = r["Arts"]["sci_weight"].get(k)
                if c is None:
                    continue
                print(f"  {lbl:<5}{c:>10.3f}{m:>10.3f}{a:>10.3f}"
                      f"{c - a:>8.3f}{a - c:>8.3f}")
            print(f"  路由 top-1 命中率 (层27): "
                  f"Code {r['Code']['router_acc'].get('27')}, "
                  f"Math {r['Math']['router_acc'].get('27')}, "
                  f"Arts {r['Arts']['router_acc'].get('27')}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")


def Path_exists(p):
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()
