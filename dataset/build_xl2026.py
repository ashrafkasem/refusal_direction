"""
Build the "2026" dataset from AIM-Intelligence/XL-SafetyBench (released April 2026, arXiv:2605.05662, CC-BY-4.0).

- harmful  : `base_query_english` from the jailbreak track (1500 unique plain harmful requests, 150 per country x 10 countries)
- harmless : `scenario_english` from the cultural track (1000 everyday benign task requests, 100 per country x 10 countries;
             the cultural track has only 200 unique `base_query_english` items, so the longer scenario requests are used)

Splits are stratified by country so every split covers all 10 countries.
Output mirrors the repo's 2023-2024 layout:
    dataset/splits_2026/{harmful,harmless}_{train,val,test}.json
    dataset/processed/xlsafety2026.json   (100 held-out harmful eval prompts, 10 per country, analogous to jailbreakbench.json)
"""
import os
import json
import random
import pandas as pd

BASE = "https://huggingface.co/datasets/AIM-Intelligence/XL-SafetyBench/resolve/main"
dataset_dir = os.path.dirname(os.path.realpath(__file__))
split_dir = os.path.join(dataset_dir, "splits_2026")
os.makedirs(split_dir, exist_ok=True)

def stratified_split(df, n_train, n_val, seed=0):
    train, val, test = [], [], []
    for _, g in df.groupby("country", sort=True):
        rows = g.to_dict("records")
        random.Random(seed).shuffle(rows)
        train += rows[:n_train]
        val += rows[n_train:n_train + n_val]
        test += rows[n_train + n_val:]
    return train, val, test

def to_records(rows, category_key, text_key="base_query_english"):
    return [{"instruction": r[text_key].strip(), "category": f'{r["country"]} / {r[category_key]}'} for r in rows]

harmful = pd.read_parquet(f"{BASE}/jailbreak.parquet").drop_duplicates("base_query_english")
harmless = pd.read_parquet(f"{BASE}/cultural.parquet").drop_duplicates("scenario_english")

# harmful: 150 per country -> 60 train / 15 val / 75 test
h_train, h_val, h_test = stratified_split(harmful, 60, 15)
# harmless: 100 per country -> 40 train / 20 val / 40 test
s_train, s_val, s_test = stratified_split(harmless, 40, 20)

splits = {
    "harmful_train": to_records(h_train, "category"),
    "harmful_val": to_records(h_val, "category"),
    "harmful_test": to_records(h_test, "category"),
    "harmless_train": to_records(s_train, "category", "scenario_english"),
    "harmless_val": to_records(s_val, "category", "scenario_english"),
    "harmless_test": to_records(s_test, "category", "scenario_english"),
}
for name, records in splits.items():
    with open(os.path.join(split_dir, f"{name}.json"), "w") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    print(name, len(records))

# held-out harmful eval set: 10 per country drawn from the harmful test split
by_country = {}
for r in h_test:
    by_country.setdefault(r["country"], []).append(r)
eval_rows = [r for c in sorted(by_country) for r in by_country[c][:10]]
with open(os.path.join(dataset_dir, "processed", "xlsafety2026.json"), "w") as f:
    json.dump(to_records(eval_rows, "category"), f, indent=4, ensure_ascii=False)
print("xlsafety2026 eval", len(eval_rows))
