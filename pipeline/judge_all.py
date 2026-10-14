"""Re-score every harmful-prompt completion file under pipeline/runs with substring matching + the local Qwen3Guard judge.

Run after generation, once Qwen/Qwen3Guard-Gen-4B is downloaded:
    python -m pipeline.judge_all
Files whose evaluation already contains Qwen3Guard scores are skipped.
"""
import os
import glob
import json

from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

RUNS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs")

for completions_path in sorted(glob.glob(os.path.join(RUNS, "*", "*", "completions", "*_completions.json"))):
    if os.path.basename(completions_path).startswith("harmless_"):
        continue  # harmless prompts are scored by refusal substring matching only, as in the original pipeline
    evaluation_path = completions_path.replace("_completions.json", "_evaluations.json")
    if os.path.exists(evaluation_path) and "qwen3guard_success_rate" in json.load(open(evaluation_path)):
        continue
    print(f"Judging {os.path.relpath(completions_path, RUNS)}")
    evaluate_jailbreak(
        completions=json.load(open(completions_path)),
        methodologies=["substring_matching", "qwen3guard"],
        evaluation_path=evaluation_path,
    )
