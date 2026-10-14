"""Quick check that the Qwen3.5 adapter works: template, module structure, refusal tokens, generation."""
import sys
import json
import torch

from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split

model_path = sys.argv[1]
load_in_4bit = "--load_in_4bit" in sys.argv

mb = construct_model_base(model_path, load_in_4bit=load_in_4bit)
print("model class:", type(mb.model).__name__)
print("layers:", len(mb.model_block_modules), "| config layers:", mb.model.config.num_hidden_layers, "| d_model:", mb.model.config.hidden_size)
print("attn module types:", sorted({type(m).__name__ for m in mb.model_attn_modules}))
print("template suffix:", repr(mb.template_suffix))
print("eoi toks:", mb.eoi_toks, [mb.tokenizer.decode([t]) for t in mb.eoi_toks])
print("refusal toks:", mb.refusal_toks, [mb.tokenizer.decode([t]) for t in mb.refusal_toks])
print("GPU memory GB:", round(torch.cuda.memory_allocated() / 1e9, 2))

harmful = load_dataset_split("harmful", "val", version="2023-2024")[:4] + load_dataset_split("harmful", "val", version="2026")[:4]
harmless = load_dataset_split("harmless", "val", version="2023-2024")[:2] + load_dataset_split("harmless", "val", version="2026")[:2]
for label, data in [("HARMFUL", harmful), ("HARMLESS", harmless)]:
    for c in mb.generate_completions(data, batch_size=8, max_new_tokens=40):
        print(f"[{label}] {c['prompt'][:70]!r}\n    -> {c['response'][:150]!r}")
