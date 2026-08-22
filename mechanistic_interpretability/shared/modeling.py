from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    layer_paths: tuple[str, ...]


MODEL_SPECS = {
    "qwen3_8b": ModelSpec(
        "qwen3_8b",
        "Qwen/Qwen3-8B",
        ("model.layers", "model.model.layers", "transformer.h"),
    ),
    "gpt_oss_20b": ModelSpec(
        "gpt_oss_20b",
        "openai/gpt-oss-20b",
        ("model.layers", "transformer.h", "model.model.layers"),
    ),
    "qwen3_30b_a3b": ModelSpec(
        "qwen3_30b_a3b",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        ("model.layers", "model.model.layers", "transformer.h"),
    ),
}


def resolve_dtype(torch: Any, value: str) -> Any:
    return {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[value]


def load_model(
    model_key: str,
    *,
    model_name: str | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str = "",
    output_router_logits: bool = False,
) -> tuple[Any, Any, Any, ModelSpec]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODEL_SPECS[model_key]
    resolved_name = model_name or spec.name
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": resolve_dtype(torch, dtype),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(resolved_name, **kwargs)
    model.eval()
    if output_router_logits and hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    input_device = model.get_input_embeddings().weight.device
    return model, tokenizer, input_device, spec


def resolve_layers(model: Any, spec: ModelSpec) -> Any:
    for path in spec.layer_paths:
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
            if len(current):
                return current
        except (AttributeError, TypeError):
            continue
    raise AttributeError(f"Could not locate transformer layers using {spec.layer_paths}")


def layer_output_tensor(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def replace_layer_output(output: Any, tensor: Any) -> Any:
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    return tensor


def render_prompt(tokenizer: Any, user_prompt: str, *, reasoning: str = "low") -> str:
    messages = [
        {
            "role": "system",
            "content": f"Reasoning: {reasoning}\nFollow the requested output format exactly.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {messages[0]['content']}\n\nUser:\n{user_prompt}\n\nAssistant:"


def last_token_addition_hook(direction: Any, alpha: float) -> Callable[[Any, Any, Any], Any]:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = layer_output_tensor(output)
        modified = hidden.clone()
        modified[:, -1, :] = modified[:, -1, :] + float(alpha) * direction.to(modified.device, modified.dtype)
        return replace_layer_output(output, modified)

    return hook
