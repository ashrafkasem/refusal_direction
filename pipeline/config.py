
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    dataset_version: str = "2023-2024" # "2023-2024" (original repo splits) or "2026" (XL-SafetyBench)
    load_in_4bit: bool = False
    n_train: int = 128
    n_test: int = 100
    n_val: int = 32
    filter_train: bool = True
    filter_val: bool = True
    # both held-out harmful sets are evaluated in every run, so each direction is also tested on the other dataset
    evaluation_datasets: Tuple[str] = ("jailbreakbench", "xlsafety2026")
    max_new_tokens: int = 256
    gen_batch_size: int = 32
    # LlamaGuard2 (Together AI API) is replaced by a local Qwen3Guard-Gen-4B judge, which runs separately
    # after generation via `python -m pipeline.judge_all` (so generation never waits on the judge model)
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching",)
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 64

    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias, self.dataset_version)
