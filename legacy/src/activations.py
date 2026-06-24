from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "Qwen/Qwen3-8B": ("qwen3-8b", "qwen3"),
    "meta-llama/Llama-3.1-8B-Instruct": ("llama3.1-8b", "llama"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("mistral-7b", "mistral"),
}


def model_family(model_id: str) -> str:
    if model_id in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_id][1]
    raise ValueError(f"Unknown model id: {model_id!r}")


def _prepare_input(
    tokenizer,
    prompt: str,
    *,
    no_think: bool = False,
    model_name: str = "",
) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if no_think and model_name and model_family(model_name) == "qwen3":
        template_kwargs["enable_thinking"] = False
    token_ids: Any = tokenizer.apply_chat_template(messages, **template_kwargs)
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _probe_get_layers(model: nn.Module) -> list[nn.Module]:
    for attr_path in ["model.layers", "model.model.layers", "transformer.h", "layers"]:
        obj = model
        found = True
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                found = False
                break
        if found and obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj)
    raise RuntimeError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        "Try adding the layer attribute path to _probe_get_layers()."
    )


def probe_setup_hooks(model: nn.Module, *, batch_size: int = 1) -> int:
    layers = _probe_get_layers(model)
    num_layers = len(layers)

    model._probe_steps: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    model._probe_hooks: list = []
    model._probe_batch_size: int = batch_size

    for layer_idx, layer in enumerate(layers):
        def _make_hook(idx: int):
            def _hook(module: nn.Module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[0] <= model._probe_batch_size:
                    model._probe_steps[idx].append(h.detach().float().cpu())
            return _hook

        handle = layer.register_forward_hook(_make_hook(layer_idx))
        model._probe_hooks.append(handle)

    return num_layers


def probe_reset(model: nn.Module) -> None:
    if hasattr(model, "_probe_steps"):
        model._probe_steps = [[] for _ in range(len(model._probe_steps))]


def probe_collect(model: nn.Module) -> torch.Tensor | None:
    if not hasattr(model, "_probe_steps"):
        return None

    result = []
    for layer_steps in model._probe_steps:
        if not layer_steps:
            return None
        layer_hidden = torch.cat(layer_steps, dim=0)
        result.append(layer_hidden)

    if not result:
        return None
    return torch.stack(result, dim=0)


def probe_remove_hooks(model: nn.Module) -> None:
    for handle in getattr(model, "_probe_hooks", []):
        handle.remove()
    for attr in ("_probe_hooks", "_probe_steps"):
        if hasattr(model, attr):
            delattr(model, attr)


def extract_generation_hidden_states_vllm(
    llm,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    no_think: bool = False,
    model_name: str = "",
    logprobs: int = 1000,
) -> tuple[str, torch.Tensor, list[int], list[dict]] | None:
    from vllm import SamplingParams

    text_input = _prepare_input(
        tokenizer, prompt, no_think=no_think, model_name=model_name
    )

    llm.apply_model(probe_reset)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
        logprobs=logprobs if logprobs > 0 else None,
    )
    outputs = llm.generate([text_input], sampling_params, use_tqdm=False)
    generated_text = outputs[0].outputs[0].text.strip()

    if not generated_text:
        return None

    hidden_states = llm.apply_model(probe_collect)[0]
    if hidden_states is None:
        return None

    token_ids: list[int] = outputs[0].outputs[0].token_ids
    logprobs_list: list[dict] = (
        outputs[0].outputs[0].logprobs if logprobs > 0 else []
    )

    return generated_text, hidden_states, token_ids, logprobs_list


def _pool_hidden_dim(acts: torch.Tensor, target_dim: int = 128) -> torch.Tensor:
    L, T, D = acts.shape
    if D == target_dim:
        return acts
    x = acts.reshape(L * T, 1, D)
    x = F.adaptive_avg_pool1d(x, target_dim)
    return x.reshape(L, T, target_dim)


def build_actmappp(
    acts: torch.Tensor,
    *,
    target_dim: int = 128,
    target_layers: int = 32,
    n_segments: int = 4,
    end_window: int = 8,
) -> torch.Tensor:
    acts_f = _pool_hidden_dim(acts.float(), target_dim)
    L, T, D = acts_f.shape

    channels: list[torch.Tensor] = []

    # Channels 0-3: means over four consecutive answer segments.
    boundaries = torch.linspace(0, T, n_segments + 1).long().tolist()
    for i in range(n_segments):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        end = max(end, start + 1)
        end = min(end, T)
        start = min(start, T - 1)
        channels.append(acts_f[:, start:end, :].mean(dim=1))

    # Channel 4: final token.
    channels.append(acts_f[:, -1, :])

    # Channel 5: mean over the last few generated tokens.
    win_start = max(0, T - end_window)
    channels.append(acts_f[:, win_start:, :].mean(dim=1))

    # Channel 6: variation across generated tokens.
    if T >= 2:
        channels.append(acts_f.std(dim=1))
    else:
        channels.append(torch.zeros(L, D, dtype=acts_f.dtype, device=acts_f.device))

    # Channel 7: peak value reached during generation.
    channels.append(acts_f.max(dim=1).values)

    # Channel 8: net change from first to last generated token.
    channels.append(acts_f[:, -1, :] - acts_f[:, 0, :])

    # Channel 9: temporal slope
    if T >= 2:
        t_idx = torch.arange(T, dtype=acts_f.dtype, device=acts_f.device)
        t_mean = t_idx.mean()
        t_var = ((t_idx - t_mean) ** 2).sum().clamp(min=1e-8)
        centered_acts = acts_f - acts_f.mean(dim=1, keepdim=True)
        centered_t = t_idx - t_mean
        channels.append(torch.einsum("ltd,t->ld", centered_acts, centered_t) / t_var)
    else:
        channels.append(torch.zeros(L, D, dtype=acts_f.dtype, device=acts_f.device))

    # Channel 10: RMS magnitude per layer.
    denom = torch.tensor(T * D, dtype=acts_f.dtype, device=acts_f.device).sqrt()
    layer_norm = acts_f.reshape(L, -1).norm(p=2, dim=1, keepdim=True) / denom.clamp(min=1e-8)
    channels.append(layer_norm.expand(L, D))

    # Channel 11: average step-to-step movement.
    if T >= 2:
        channels.append((acts_f[:, 1:, :] - acts_f[:, :-1, :]).abs().mean(dim=1))
    else:
        channels.append(torch.zeros(L, D, dtype=acts_f.dtype, device=acts_f.device))

    stacked = torch.stack(channels, dim=0)
    stacked_2d = stacked.permute(0, 2, 1)
    stacked_2d = F.adaptive_avg_pool1d(stacked_2d, target_layers)
    return stacked_2d.permute(0, 2, 1)


def normalize_actmap(
    actmap: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if actmap.ndim != 3:
        raise ValueError(f"Expected [C, L, D] ActMap, got shape {tuple(actmap.shape)}")

    actmap_f = actmap.float()
    mean = actmap_f.mean(dim=(1, 2), keepdim=True)
    std = actmap_f.std(dim=(1, 2), keepdim=True).clamp(min=eps)
    return (actmap_f - mean) / std
