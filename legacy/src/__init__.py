from __future__ import annotations

_ACTIVATION_EXPORTS = {
    "build_actmappp",
    "extract_generation_hidden_states_vllm",
    "normalize_actmap",
    "probe_setup_hooks",
}

_EVALUATION_EXPORTS = {
    "clean_prediction",
    "evaluate_gsm8k_answer",
    "evaluate_nq_open_answer",
    "evaluate_triviaqa_answer",
    "evaluate_web_questions_answer",
    "extract_gsm8k_gold_answer",
    "extract_gsm8k_numeric_answer",
    "format_gsm8k_prompt",
    "format_nq_open_prompt",
    "format_triviaqa_prompt",
    "format_web_questions_prompt",
    "normalize_answer",
    "parse_aliases",
    "token_f1",
}

__all__ = sorted(_ACTIVATION_EXPORTS | _EVALUATION_EXPORTS)


def __getattr__(name: str):
    if name in _ACTIVATION_EXPORTS:
        from . import activations

        return getattr(activations, name)
    if name in _EVALUATION_EXPORTS:
        from . import evaluation

        return getattr(evaluation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
