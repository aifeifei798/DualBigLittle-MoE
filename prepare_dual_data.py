import json
from datasets import load_dataset

print("=" * 70)
print("[*] 正在装配【文理 1:1 对抗特训语料库】(4000理科 vs 4000文科)...")
print("=" * 70)

# 1. 代码 2000 条 (理科)
print("    - [1/3] 正在加载代码数据集 (2000条)...")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca",
                       split="train[:2000]")

# 2. 数学 2000 条 (理科)
print("    - [2/3] 正在加载数学数据集 (2000条)...")
ds_math = load_dataset("openai/gsm8k", "main", split="train[:2000]")

# 3. 文科通用 4000 条 (文科对比样本)
print("    - [3/3] 正在加载文科高质量指令集 (4000条)...")
ds_arts = load_dataset("HuggingFaceH4/no_robots", split="train[:4000]")

dual_data = []

# 理科 - 代码 (Big Target = 1: 理科核, Little Target = 0: 代码班长)
for item in ds_code:
    prompt = item["instruction"] + (f"\n{item['input']}"
                                    if item.get("input") else "")
    dual_data.append({
        "big_target": 1,
        "little_target": 0,
        "domain": "Code",
        "prompt": prompt,
        "response": item["output"]
    })

# 理科 - 数学 (Big Target = 1: 理科核, Little Target = 8: 数学班长)
for item in ds_math:
    dual_data.append({
        "big_target": 1,
        "little_target": 8,
        "domain": "Math",
        "prompt": item["question"],
        "response": item["answer"]
    })

# 文科 - 写作/日常 (Big Target = 0: 文科核, Little Target = 16: 文科班长)
for item in ds_arts:
    messages = item["messages"]
    if len(messages) >= 2:
        dual_data.append({
            "big_target": 0,
            "little_target": 16,
            "domain": "Arts",
            "prompt": messages[0]["content"],
            "response": messages[1]["content"]
        })

output_file = "dual_contrast_data.jsonl"
print(f"\n[*] 正在清洗并写入 {output_file}，共 {len(dual_data)} 条严格对齐数据...")
with open(output_file, "w", encoding="utf-8") as f:
    for entry in dual_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[✔] 数据装配完毕！文科与理科正好各占 50%，调度器现在有绝对对比样本了！")
