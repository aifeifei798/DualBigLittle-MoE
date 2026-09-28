"""语料加载、标签掩码与 schema 处理的测试。"""

from __future__ import annotations

import json

import pytest
import torch

from dbl.config import Config
from dbl.data import DualContrastDataset, load_jsonl, resolve_group
from dbl.prepare import clean_gsm8k, dedupe, split_by_domain


class FakeTokenizer:
    """按空白切分的假 tokenizer，行为足够验证掩码逻辑。"""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}


@pytest.fixture
def cfg():
    return Config(
        model_id="dummy", num_experts=8, top_k=2, group_sizes=(2, 2, 4),
        max_length=16, device="cpu", dtype="float32",
    )


@pytest.fixture
def rows():
    return [
        {"big_target": 1, "little_group": 0, "domain": "Code",
         "prompt": "a b c", "response": "d e f"},
        {"big_target": 1, "little_group": 1, "domain": "Math",
         "prompt": "a b", "response": "c d e f g"},
        {"big_target": 0, "little_group": 2, "domain": "Arts",
         "prompt": "x y", "response": "z"},
    ]


@pytest.fixture
def dataset(tmp_path, rows, cfg):
    p = tmp_path / "d.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return DualContrastDataset(p, FakeTokenizer(), cfg)


class TestLoadJsonl:
    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_text('{"a":1}\n\n  \n{"a":2}\n', encoding="utf-8")
        assert len(load_jsonl(p)) == 2

    def test_reports_bad_line_number(self, tmp_path):
        p = tmp_path / "b.jsonl"
        p.write_text('{"a":1}\nnot-json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="b.jsonl:2"):
            load_jsonl(p)


class TestResolveGroup:
    """schema 漂移的兼容层：历史语料只有 little_target。"""

    def test_prefers_little_group(self, cfg):
        item = {"little_group": 2, "little_target": 0, "domain": "Code"}
        assert resolve_group(item, cfg.groups) == 2

    def test_falls_back_to_domain(self, cfg):
        assert resolve_group({"domain": "Math"}, cfg.groups) == 1
        assert resolve_group({"domain": "Arts"}, cfg.groups) == 2

    def test_falls_back_to_legacy_little_target(self):
        """旧语料的 little_target 存的是分组**起始下标**。

        用真实的 32 专家拓扑验证：0 -> 代码组，8 -> 数学组，16 -> 写作组。
        """
        g = Config().groups          # 默认 8/8/16
        assert resolve_group({"little_target": 0}, g) == 0
        assert resolve_group({"little_target": 8}, g) == 1
        assert resolve_group({"little_target": 16}, g) == 2

    def test_legacy_index_anywhere_in_group(self, cfg):
        """组内任意专家下标都应映射回该组（cfg 为 2/2/4 拓扑）。"""
        assert resolve_group({"little_target": 0}, cfg.groups) == 0
        assert resolve_group({"little_target": 1}, cfg.groups) == 0
        assert resolve_group({"little_target": 2}, cfg.groups) == 1
        assert resolve_group({"little_target": 3}, cfg.groups) == 1
        assert resolve_group({"little_target": 4}, cfg.groups) == 2
        assert resolve_group({"little_target": 7}, cfg.groups) == 2

    def test_legacy_value_outside_any_group(self, cfg):
        with pytest.raises(ValueError, match="不落在任何分组"):
            resolve_group({"little_target": 99}, cfg.groups)


class TestLabelMasking:
    def _ntok(self, text):
        return len(FakeTokenizer()(text)["input_ids"])

    def test_prompt_is_masked(self, dataset):
        item = dataset[0]
        labels = item["labels"].tolist()
        n_prompt = self._ntok(
            "<|im_start|>user\na b c<|im_end|>\n<|im_start|>assistant\n"
        )
        n_full = self._ntok(
            "<|im_start|>user\na b c<|im_end|>\n"
            "<|im_start|>assistant\nd e f<|im_end|>\n"
        )
        assert labels[:n_prompt] == [-100] * n_prompt
        # response 段被监督
        assert labels[n_prompt:n_full] == [1] * (n_full - n_prompt)
        assert n_full - n_prompt == 3

    def test_padding_is_masked(self, dataset):
        item = dataset[0]
        n_real = int(item["attention_mask"].sum())
        assert item["attention_mask"].tolist() == [1] * n_real + [0] * (
            16 - n_real
        )
        labels = item["labels"].tolist()
        assert labels[n_real:] == [-100] * (16 - n_real)
        assert all(v == -100 for v in labels[8:])   # 尾部无监督

    def test_shapes_and_dtypes(self, dataset):
        item = dataset[0]
        for k in ("input_ids", "attention_mask", "labels"):
            assert item[k].shape == (16,)
            assert item[k].dtype == torch.long
        assert item["big_target"].dtype == torch.long
        assert item["group_target"].dtype == torch.long

    def test_truncation_flagged(self, tmp_path, cfg):
        """超长样本应被标记，而不是静默丢弃 response。"""
        row = {"big_target": 1, "little_group": 0, "domain": "Code",
               "prompt": " ".join(f"p{i}" for i in range(20)),
               "response": " ".join(f"r{i}" for i in range(20))}
        p = tmp_path / "long.jsonl"
        p.write_text(json.dumps(row) + "\n", encoding="utf-8")
        d = DualContrastDataset(p, FakeTokenizer(), cfg)
        assert bool(d[0]["truncated"])
        assert d[0]["labels"].shape[0] == cfg.max_length

    def test_labels_never_exceed_max_length(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i]["labels"].shape[0] == dataset.cfg.max_length

    def test_group_target_present_in_schema(self, dataset):
        assert dataset[0]["group_target"].item() == 0
        assert dataset[1]["group_target"].item() == 1
        assert dataset[2]["group_target"].item() == 2


class TestSchemaValidation:
    def test_missing_field_raises_with_hint(self, tmp_path, cfg):
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps({"prompt": "a", "response": "b"}) + "\n",
                     encoding="utf-8")
        with pytest.raises(ValueError, match="big_target"):
            DualContrastDataset(p, FakeTokenizer(), cfg)

    def test_requires_pad_token(self, tmp_path, rows, cfg):
        class NoPad(FakeTokenizer):
            pad_token_id = None

        p = tmp_path / "d.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        with pytest.raises(ValueError, match="pad_token"):
            DualContrastDataset(p, NoPad(), cfg)


class TestGsm8kCleaning:
    def test_strips_calculator_annotation(self):
        raw = "He has 3 apples.\nHe buys 4 more.\n#### 7"
        out = clean_gsm8k(raw)
        assert "####" not in out
        assert out.endswith("The answer is 7.")

    def test_keeps_reasoning_body(self):
        raw = "Step one.\nStep two.\n#### 42"
        out = clean_gsm8k(raw)
        assert "Step one." in out
        assert "Step two." in out

    def test_handles_answer_without_annotation(self):
        assert clean_gsm8k("Just an answer.") == "Just an answer."

    def test_takes_last_annotation(self):
        raw = "work\n#### 1\nmore work\n#### 99"
        out = clean_gsm8k(raw)
        assert "####" not in out
        assert out.endswith("The answer is 99.")

    def test_strips_trailing_period_from_number(self):
        assert clean_gsm8k("work\n#### 7.").endswith("The answer is 7.")


class TestDedupe:
    def test_removes_exact_duplicates(self, rows):
        out, dropped = dedupe(rows + [rows[0]])
        assert dropped == 1
        assert len(out) == len(rows)

    def test_keeps_distinct(self, rows):
        out, dropped = dedupe(rows)
        assert dropped == 0
        assert len(out) == 3


class TestSplitByDomain:
    def test_stratified_by_domain(self):
        rows = (
            [{"domain": "Code", "prompt": str(i), "response": "r"} for i in range(20)]
            + [{"domain": "Arts", "prompt": str(i), "response": "r"} for i in range(20)]
        )
        tr, va = split_by_domain(rows, 0.2, seed=0)
        tr_c = sum(1 for r in tr if r["domain"] == "Code")
        va_c = sum(1 for r in va if r["domain"] == "Code")
        assert va_c == 4
        assert tr_c == 16
        assert len(tr) + len(va) == 40

    def test_deterministic_given_seed(self):
        rows = [{"domain": "Code", "prompt": str(i), "response": "r"}
                for i in range(50)]
        a = split_by_domain(rows, 0.2, seed=7)
        b = split_by_domain(rows, 0.2, seed=7)
        assert a[0] == b[0] and a[1] == b[1]

    def test_no_overlap(self):
        rows = [{"domain": "Code", "prompt": str(i), "response": "r"}
                for i in range(30)]
        tr, va = split_by_domain(rows, 0.3, seed=1)
        tr_p = {r["prompt"] for r in tr}
        va_p = {r["prompt"] for r in va}
        assert not (tr_p & va_p)
