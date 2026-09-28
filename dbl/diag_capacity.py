"""验证设计冲突推论。

推论：Tier-1 冻结，因此文科无法从『路由到文科核』获益 —— 那样等于
退化为原版模型。文科的收益只能来自 Tier-2 持续参与。

若推论成立，提高 ``lr_big_sci``（让 Tier-2 学更多）应当**同时**改善
文科与理科，而与路由是否锐利无关。

对照实验（其余相同）：
  基线    lr_big_sci=2e-5, aux_w=0.1  (v5 配方)
  提容量  lr_big_sci=1e-4, aux_w=0.1
  弱路由  lr_big_sci=2e-5, aux_w=0.0036 (v3 配方)
  弱路由+提容量 lr_big_sci=1e-4, aux_w=0.0036
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dbl.checkpoint import apply_checkpoint, load_checkpoint
from dbl.config import Config
from dbl.data import DualContrastDataset
from dbl.moe import inject_moe
from dbl.train import train

DEV = "cuda:0"
DOMAINS = ("Code", "Math", "Arts")
VAL = "dual_contrast_val.jsonl"


@torch.no_grad()
def eval_ppl(weights, n=100, bs=8):
    payload = load_checkpoint(weights, map_location="cpu")
    cfg = payload.get("cfg") or Config()
    cfg = cfg.replace(device=DEV)
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.pad_token = tok.eos_token
    ds = DualContrastDataset(VAL, tok, cfg)

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
    ntok = 0
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
            for m in mods:
                p = torch.softmax(m.last_stats.big_logits[:, -1, :].float(), -1)
                sci[d] += float(p[:, 1].sum())
                ntok += p.shape[0]

    out = {d: round(float(torch.tensor(nll[d] / tokc[d]).exp()), 2)
           for d in DOMAINS}
    sciw = {d: round(sci[d] / max(ntok, 1), 3) for d in DOMAINS}
    del model
    torch.cuda.empty_cache()
    return out, sciw


def main():
    runs = [
        ("基线      lr_sci=2e-5 aux=0.1",   2e-5, 0.1),
        ("提容量    lr_sci=1e-4 aux=0.1",   1e-4, 0.1),
        ("弱路由    lr_sci=2e-5 aux=0.0036", 2e-5, 0.0036),
        ("弱路由+容量 lr_sci=1e-4 aux=0.0036", 1e-4, 0.0036),
    ]
    d = Path(tempfile.mkdtemp())
    print("=" * 76)
    print("验证：Tier-2 容量是否同时改善文理两域")
    print("=" * 76)
    rows = []
    for tag, lr_sci, aux in runs:
        cfg = Config(lr_big_sci=lr_sci, router_aux_weight=aux)
        ck = d / f"{lr_sci}_{aux}.pt"
        cfg = cfg.replace(weights_path=str(ck))
        info = train(cfg, log_every=10**9, save_every=10**9,
                     metrics_path=str(d / f"{lr_sci}_{aux}.jsonl"))
        ppl, sciw = eval_ppl(info["path"])
        rows.append((tag, ppl, sciw))
        print(f"  {tag:<38} PPL Code {ppl['Code']:>6.2f}  "
              f"Math {ppl['Math']:>6.2f}  Arts {ppl['Arts']:>6.2f}   "
              f"Arts w_sci={sciw['Arts']:.3f}")

    print()
    print("=" * 76)
    print("Arts PPL 随 lr_big_sci 的变化（其余相同）")
    print("=" * 76)
    print(f"  aux=0.1     : lr 2e-5 -> {rows[0][1]['Arts']:>6.2f} , "
          f"lr 1e-4 -> {rows[1][1]['Arts']:>6.2f}")
    print(f"  aux=0.0036  : lr 2e-5 -> {rows[2][1]['Arts']:>6.2f} , "
          f"lr 1e-4 -> {rows[3][1]['Arts']:>6.2f}")


if __name__ == "__main__":
    main()
