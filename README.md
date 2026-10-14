# Refusal in Language Models Is Mediated by a Single Direction

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains code and results accompanying the paper "Refusal in Language Models Is Mediated by a Single Direction".
In the spirit of scientific reproducibility, we provide code to reproduce the main results from the paper.

- [Paper](https://arxiv.org/abs/2406.11717)
- [Blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## Setup

```bash
git clone https://github.com/andyrdt/refusal_direction.git
cd refusal_direction
source setup.sh
```

The setup script will prompt you for a HuggingFace token (required to access gated models) and a Together AI token (required to access the Together AI API, which is used for evaluating jailbreak safety scores).
It will then set up a virtual environment and install the required packages.

## Reproducing main results

To reproduce the main results from the paper, run the following command:

```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```
where `{model_path}` is the path to a HuggingFace model. For example, for Llama-3 8B Instruct, the model path would be `meta-llama/Meta-Llama-3-8B-Instruct`.

The pipeline performs the following steps:
1. Extract candiate refusal directions
    - Artifacts will be saved in `pipeline/runs/{model_alias}/generate_directions`
2. Select the most effective refusal direction
    - Artifacts will be saved in `pipeline/runs/{model_alias}/select_direction`
    - The selected refusal direction will be saved as `pipeline/runs/{model_alias}/direction.pt`
3. Generate completions over harmful prompts, and evaluate refusal metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/completions`
4. Generate completions over harmless prompts, and evaluate refusal metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/completions`
5. Evaluate CE loss metrics.
    - Artifacts will be saved in `pipeline/runs/{model_alias}/loss_evals`

For convenience, we have included pipeline artifacts for the smallest model in each model family:
- [`qwen/qwen-1_8b-chat`](/pipeline/runs/qwen-1_8b-chat/)
- [`google/gemma-2b-it`](/pipeline/runs/gemma-2b-it/)
- [`01-ai/yi-6b-chat`](/pipeline/runs/yi-6b-chat/)
- [`meta-llama/llama-2-7b-chat-hf`](/pipeline/runs/llama-2-7b-chat-hf/)
- [`meta-llama/meta-llama-3-8b-instruct`](/pipeline/runs/meta-llama-3-8b-instruct/)

## Dataset versions and Qwen3.5 support (this fork)

The pipeline can extract the refusal direction from either of two harmful/harmless datasets:

| `--dataset_version` | Harmful prompts | Harmless prompts | Splits |
|---|---|---|---|
| `2023-2024` (default) | AdvBench, MaliciousInstruct, TDC2023, HarmBench, JailbreakBench, StrongREJECT | Alpaca | `dataset/splits/` (original) |
| `2026` | [XL-SafetyBench](https://huggingface.co/datasets/AIM-Intelligence/XL-SafetyBench) (Apr 2026) jailbreak-track base queries, 1,500 prompts across 10 countries | XL-SafetyBench cultural-track scenarios, 999 prompts | `dataset/splits_2026/` |

The 2026 splits are stratified by country and are rebuilt with `python dataset/build_xl2026.py`. They share no prompts with the 2023-2024 data. Every run evaluates on both held-out harmful sets (`jailbreakbench` and `xlsafety2026`), so each direction is also tested on the other dataset.

Other changes:
- `pipeline/model_utils/qwen35_model.py` adds Qwen3.5 (hybrid Gated-DeltaNet / full-attention) models in non-thinking mode, with optional 4-bit NF4 loading (`--load_in_4bit`).
- The LlamaGuard2 judge (Together AI API) is replaced by a local [Qwen3Guard-Gen-4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) judge, run after generation with `python -m pipeline.judge_all`.
- `python -m pipeline.summarize` collects all runs into `pipeline/runs/summary.json`.

```bash
uv sync                                   # creates .venv (CUDA 13.0 torch on Linux and Windows)
python -m pipeline.run_pipeline --model_path Qwen/Qwen3.5-4B --dataset_version 2026 --gen_batch_size 16
python -m pipeline.run_pipeline --model_path Qwen/Qwen3.5-9B --dataset_version 2026 --gen_batch_size 16 --load_in_4bit
./master.sh                               # full experiment: 2 models x 2 datasets, judging, summary
```

## Minimal demo Colab

As part of our [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), we included a minimal demo of bypassing refusal. This demo is available as a [Colab notebook](https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw).

## As featured in

Since publishing our initial [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) in April 2024, our methodology has been independently reproduced and used many times. In particular, we acknowledge [Fail](https://huggingface.co/failspy)[Spy](https://x.com/failspy) for their work in reproducing and extending our methodology.

Our work has been featured in:
- [HackerNews](https://news.ycombinator.com/item?id=40242939)
- [Last Week in AI podcast](https://open.spotify.com/episode/2E3Fc50GVfPpBvJUmEwlOU)
- [Llama 3 hackathon](https://x.com/AlexReibman/status/1789895080754491686)
- [Applying refusal-vector ablation to a Llama 3 70B agent](https://www.lesswrong.com/posts/Lgq2DcuahKmLktDvC/applying-refusal-vector-ablation-to-a-llama-3-70b-agent)
- [Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration)


## Citing this work

If you find this work useful in your research, please consider citing our [paper](https://arxiv.org/abs/2406.11717):
```tex
@article{arditi2024refusal,
  title={Refusal in Language Models Is Mediated by a Single Direction},
  author={Andy Arditi and Oscar Obeso and Aaquib Syed and Daniel Paleka and Nina Panickssery and Wes Gurnee and Neel Nanda},
  journal={arXiv preprint arXiv:2406.11717},
  year={2024}
}
```