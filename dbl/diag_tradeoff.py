"""决定性对照：v3 式的『弱路由』与 v5 式的『强路由』在 PPL 上差在哪。

关键问题：v3 的 router 停在随机、几乎不用 Tier-2（Arts 权重≈0.25），
PPL 却好 40%；v5 的 router 正常分化，Arts 权重被压到 0.20，PPL 只好 10%。

假设：v3 的优势不来自路由，而来自**训练步数/epoch 的偶然性**，
或者来自 big_sci 在 router 随机时被更充分地利用。
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dbl.checkpoint import apply_checkpoint, load_checkpoint
from dbl.config import Config
from dbl.data import DualContrastDataset
from dbl.moe import inject_moe
from dbl.runtime import format_report, resolve_device

#: 诊断脚本统一走设备解析（auto / cpu / cuda:N），不再硬编码卡号。
#: 刻意保持为 "auto" 字面量而非导入期解析结果——否则本模块在没有
#: 显卡的机器上连 import 都会失败，无法 --help、无法被测试导入。
DEV = "auto"
DOMAINS = ("Code", "Math", "Arts")


@torch.no_grad()
def eval_ppl(weights, data, n=100, bs=8, router_aux=None):
    payload = load_checkpoint(weights, map_location="cpu")
    cfg = payload.get("cfg") or Config()
    cfg = cfg.replace(device=DEV)
    if router_aux is not None:
        cfg = cfg.replace(router_aux_weight=router_aux)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.pad_token = tok.eos_token
    ds = DualContrastDataset(data, tok, cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=cfg.torch_dtype, device_map=DEV
    )
    for p in model.parameters():
        p.requires_grad = False
    mods = inject_moe(model, cfg, mode="train")
    apply_checkpoint(mods, payload)
    model.eval()
    for m in mods:
        m.collect_stats = True

    import torch.nn.functional as F

    by_dom: dict[str, list[int]] = {}
    for i, r in enumerate(ds.rows):
        by_dom.setdefault(r["domain"], []).append(i)

    nll = {d: 0.0 for d in DOMAINS}
    tokc = {d: 0 for d in DOMAINS}
    sci = {d: 0.0 for d in DOMAINS}
    ntok = {d: 0 for d in DOMAINS}
    for d in DOMAINS:
        idx = by_dom[d][:n]
        for s in range(0, len(idx), bs):
            chunk = idx[s : s + bs]
            items = [ds[i] for i in chunk]
            ids = torch.stack([it["input_ids"] for it in items]).to(DEV)
            am = torch.stack([it["attention_mask"] for it in items]).to(DEV)
            lab = torch.stack([it["labels"] for it in items]).to(DEV)
            lg = model(input_ids=ids, attention_mask=am).logits
            sel = lab[:, 1:] != -100
            nll[d] += F.cross_entropy(
                lg[:, :-1, :][sel].float(), lab[:, 1:][sel], reduction="sum"
            ).item()
            tokc[d] += int(sel.sum())
            # Tier-2 平均权重：取末 token 的路由（近似推理口径）
            for m in mods:
                p = torch.softmax(
                    m.last_stats.big_logits[:, -1, :].float(), -1
                )
                sci[d] += float(p[:, 1].sum())
                ntok[d] += p.shape[0]

    out = {}
    for d in DOMAINS:
        out[d] = {
            "ppl": round(float(torch.tensor(nll[d] / tokc[d]).exp()), 3),
            "sci_weight": round(sci[d] / max(ntok[d], 1), 3),
        }
    del model
    torch.cuda.empty_cache()
    return out


def main():
    global DEV
    DEV = resolve_device(DEV, allow_cpu=False)
    print(format_report())
    print("=" * 74)
    print("v3(弱路由) vs v5(强路由) 逐域分解")
    print("=" * 74)
    val = "dual_contrast_val.jsonl"
    for tag, w in [("v3 弱路由", "checkpoints/v3_full.pt"),
                   ("v5 强路由", "checkpoints/v5_auxfix.pt")]:
        r = eval_ppl(w, val)
        print(f"\n  {tag}  ({w})")
        for d in DOMAINS:
            print(f"    {d:<5} PPL {r[d]['ppl']:>7.2f}   "
                  f"末token 理科核权重 {r[d]['sci_weight']:.3f}")

    # 关键对照：把 v3 的权重用**强 aux 的 config** 加载，验证差异来自权重而非 config
    print()
    print("=" * 74)
    print("对照：确认差异来自训练结果，而非 checkpoint 里的 config")
    print("=" * 74)
    r = eval_ppl("checkpoints/v3_full.pt", val, router_aux=0.1)
    print("  v3 权重 + aux_w=0.1 覆盖:")
    for d in DOMAINS:
        print(f"    {d:<5} PPL {r[d]['ppl']:>7.2f}")


if __name__ == "__main__":
    main()
