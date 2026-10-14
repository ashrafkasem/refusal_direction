"""Collect results from pipeline/runs/{model}/{dataset_version}/ into pipeline/runs/summary.json."""
import os
import json
import glob

RUNS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs")
VERSIONS = ["2023-2024", "2026"]


def load(path):
    with open(path) as f:
        return json.load(f)


def summarize_run(run_dir):
    out = {}
    meta = os.path.join(run_dir, "direction_metadata.json")
    if not os.path.exists(meta):
        return None
    out["direction"] = load(meta)

    filtered = os.path.join(run_dir, "select_direction", "direction_evaluations_filtered.json")
    if os.path.exists(filtered):
        best = [d for d in load(filtered) if d["position"] == out["direction"]["pos"] and d["layer"] == out["direction"]["layer"]]
        if best:
            out["direction"].update({k: best[0][k] for k in ("refusal_score", "steering_score", "kl_div_score")})

    out["evals"] = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "completions", "*_evaluations.json"))):
        name = os.path.basename(f).replace("_evaluations.json", "")
        dataset, intervention = name.rsplit("_", 1)
        ev = load(f)
        entry = {k: v for k, v in ev.items() if k.endswith("_rate")}
        if dataset == "harmless":
            # on harmless prompts the metric of interest is the refusal rate (= 1 - substring "jailbreak" rate)
            entry = {"refusal_rate_substring": 1 - ev["substring_matching_success_rate"]}
        out["evals"].setdefault(dataset, {})[intervention] = entry

    out["loss"] = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "loss_evals", "*_loss_eval.json"))):
        out["loss"][os.path.basename(f).replace("_loss_eval.json", "")] = load(f)
    return out


def main():
    summary = {}
    for model_dir in sorted(glob.glob(os.path.join(RUNS, "*"))):
        for version in VERSIONS:
            run_dir = os.path.join(model_dir, version)
            if os.path.isdir(run_dir):
                s = summarize_run(run_dir)
                if s is not None:
                    summary.setdefault(os.path.basename(model_dir), {})[version] = s
    with open(os.path.join(RUNS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
