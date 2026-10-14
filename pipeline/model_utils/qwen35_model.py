import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Qwen3.5 (Feb 2026) small models: hybrid Gated-DeltaNet ("linear_attention") + full-attention decoder layers.
# The chat template is the Qwen3 ChatML template; we use non-thinking mode (empty <think></think> block)
# so that the first generated token is the start of the actual answer, as the refusal-score metric assumes.
INSTRUCTION_PLACEHOLDER = "<<<INSTRUCTION>>>"

# First tokens of Qwen3.5 refusals ("I can't...", "I'm sorry...", "Sorry, ..."); set from observed baseline generations.
QWEN35_REFUSAL_STRS = ["I", "Sorry", "As"]


def _split_chat_template(tokenizer):
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION_PLACEHOLDER}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix, suffix = templated.split(INSTRUCTION_PLACEHOLDER)
    return prefix, suffix


def tokenize_instructions_qwen35_chat(
    tokenizer: AutoTokenizer,
    template_prefix: str,
    template_suffix: str,
    instructions: List[str],
    outputs: List[str]=None,
    include_trailing_whitespace=True,
):
    suffix = template_suffix if include_trailing_whitespace else template_suffix.rstrip()
    prompts = [template_prefix + instruction + suffix for instruction in instructions]
    if outputs is not None:
        prompts = [prompt + output for prompt, output in zip(prompts, outputs)]

    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt", add_special_tokens=False)


def _layer_attn(layer):
    return layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn


def orthogonalize_qwen35_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for layer in model.model.layers:
        attn = _layer_attn(layer)
        out_proj = attn.o_proj if hasattr(attn, "o_proj") else attn.out_proj
        out_proj.weight.data = get_orthogonalized_matrix(out_proj.weight.data.T, direction).T
        layer.mlp.down_proj.weight.data = get_orthogonalized_matrix(layer.mlp.down_proj.weight.data.T, direction).T


def act_add_qwen35_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    down_proj = model.model.layers[layer-1].mlp.down_proj
    bias = (coeff * direction).to(dtype=down_proj.weight.dtype, device=down_proj.weight.device)
    down_proj.bias = torch.nn.Parameter(bias)


class Qwen35Model(ModelBase):

    def __init__(self, model_name_or_path: str, load_in_4bit: bool=False):
        self.load_in_4bit = load_in_4bit
        super().__init__(model_name_or_path)

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model_kwargs = {}
        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # AutoModelForCausalLM resolves the Qwen3.5 checkpoint to its text-only causal LM (vision tower is not loaded)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="cuda",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.template_prefix, self.template_suffix = _split_chat_template(tokenizer)
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen35_chat,
            tokenizer=self.tokenizer,
            template_prefix=self.template_prefix,
            template_suffix=self.template_suffix,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(self.template_suffix, add_special_tokens=False)

    def _get_refusal_toks(self):
        toks = []
        for s in QWEN35_REFUSAL_STRS:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            assert len(ids) == 1, f"refusal string {s!r} is not a single token: {ids}"
            toks.append(ids[0])
        return toks

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([_layer_attn(layer) for layer in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([layer.mlp for layer in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen35_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen35_weights, direction=direction, coeff=coeff, layer=layer)
