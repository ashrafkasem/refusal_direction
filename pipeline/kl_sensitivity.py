"""Supplementary check: relax the paper's KL < 0.1 filter to KL < 1.0 and ablate the strongest remaining candidate.

Used when the standard selection (run_pipeline) falls back to a weak direction. Writes completions/evaluations to
pipeline/runs/<model>/<version>-kl-relaxed/ and does not touch the standard run.

    python -m pipeline.kl_sensitivity --model_path Qwen/Qwen3.5-9B --dataset_version 2026 --load_in_4bit
"""
import os
import gc
import json
import argparse

import torch

from dataset.load_dataset import load_dataset
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

RUNS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_version", default="2026")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--gen_batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--kl_threshold", type=float, default=1.0)
    args = ap.parse_args()

    alias = os.path.basename(args.model_path) + ("-nf4" if args.load_in_4bit else "")
    src = os.path.join(RUNS, alias, args.dataset_version)
    out = os.path.join(RUNS, alias, f"{args.dataset_version}-kl-relaxed")
    os.makedirs(os.path.join(out, "completions"), exist_ok=True)

    n_layer = None
    scores = json.load(open(os.path.join(src, "select_direction", "direction_evaluations.json")))
    n_layer = max(s["layer"] for s in scores) + 1
    # same filters as the paper, but KL < 1.0 instead of 0.1 (an unbounded KL admits candidates that break the model)
    ok = [s for s in scores if s["layer"] < int(n_layer * 0.8) and s["steering_score"] >= 0 and s["kl_div_score"] < args.kl_threshold]
    best = min(ok, key=lambda s: s["refusal_score"])
    print(f"KL-relaxed choice: pos={best['position']} layer={best['layer']} refusal={best['refusal_score']:.3f} "
          f"steering={best['steering_score']:.3f} kl={best['kl_div_score']:.3f}")

    mean_diffs = torch.load(os.path.join(src, "generate_directions", "mean_diffs.pt"))
    direction = mean_diffs[best["position"], best["layer"]]
    torch.save(direction, os.path.join(out, "direction.pt"))
    json.dump({"pos": best["position"], "layer": best["layer"], "refusal_score": best["refusal_score"],
               "steering_score": best["steering_score"], "kl_div_score": best["kl_div_score"],
               "note": f"strongest candidate passing the paper's filters with KL < {args.kl_threshold} instead of 0.1"},
              open(os.path.join(out, "direction_metadata.json"), "w"), indent=4)

    mb = construct_model_base(args.model_path, load_in_4bit=args.load_in_4bit)
    pre, post = get_all_direction_ablation_hooks(mb, direction)
    for ds in ["jailbreakbench", "xlsafety2026"]:
        path = os.path.join(out, "completions", f"{ds}_ablation_completions.json")
        if not os.path.exists(path):
            comps = mb.generate_completions(load_dataset(ds), fwd_pre_hooks=pre, fwd_hooks=post,
                                            max_new_tokens=args.max_new_tokens, batch_size=args.gen_batch_size)
            json.dump(comps, open(path, "w"), indent=4)
    del pre, post
    mb.del_model(); del mb; gc.collect(); torch.cuda.empty_cache()

    for ds in ["jailbreakbench", "xlsafety2026"]:
        comps = json.load(open(os.path.join(out, "completions", f"{ds}_ablation_completions.json")))
        evaluate_jailbreak(completions=comps, methodologies=["substring_matching", "qwen3guard"],
                           evaluation_path=os.path.join(out, "completions", f"{ds}_ablation_evaluations.json"))


if __name__ == "__main__":
    main()
