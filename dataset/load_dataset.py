import os
import json

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

# "2023-2024": the original splits shipped with the repo (AdvBench, MaliciousInstruct, TDC2023, HarmBench, JailbreakBench, StrongREJECT / Alpaca)
# "2026": XL-SafetyBench (April 2026), built by dataset/build_xl2026.py
DATASET_VERSIONS = {
    '2023-2024': 'splits',
    '2026': 'splits_2026',
}

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca", "xlsafety2026"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False, version: str='2023-2024'):
    assert harmtype in HARMTYPES
    assert split in SPLITS
    assert version in DATASET_VERSIONS, f"Valid versions: {list(DATASET_VERSIONS)}"

    file_path = os.path.join(dataset_dir_path, DATASET_VERSIONS[version], f'{harmtype}_{split}.json')

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset
