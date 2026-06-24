from __future__ import annotations

import ast
import re
import string
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any


TRIVIAQA_PROMPT_TEMPLATE = (
    "Answer the trivia question with the shortest correct answer. "
    "Output only the answer, not a sentence.\n\n"
    "Question: {question}\n"
    "Answer:"
)

NQ_OPEN_PROMPT_TEMPLATE = (
    "Answer the question with a short, factual answer. "
    "Output only the answer, not a sentence.\n\n"
    "Question: {question}\n"
    "Answer:"
)

WEBQUESTIONS_PROMPT_TEMPLATE = (
    "Answer the question with a short, factual answer. "
    "Output only the answer, not a sentence.\n\n"
    "Question: {question}\n"
    "Answer:"
)

GSM8K_PROMPT_TEMPLATE = (
    "Solve the grade-school math word problem. "
    "Return only the final numeric answer, without units or explanation.\n\n"
    "Question: {question}\n"
    "Answer:"
)

_NUMBER_RE_TEXT = (
    r"(?<![\w])"
    r"[-+]?"
    r"(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)?"
    r"(?![\w])"
)
_GSM8K_NUMBER_RE = re.compile(_NUMBER_RE_TEXT)
_GSM8K_HASH_RE = re.compile(r"####\s*(?P<number>" + _NUMBER_RE_TEXT + r")", re.IGNORECASE)
_GSM8K_DIRECT_ANSWER_RE = re.compile(
    r"\b(?:final\s+answer|answer|result|total)\b"
    r"\s*(?:is|are|=|:)\s*"
    r"[$]?\s*(?P<number>" + _NUMBER_RE_TEXT + r")",
    re.IGNORECASE,
)
_GSM8K_LINE_ANSWER_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:=]\s*(?P<body>.+)$", re.IGNORECASE)
_GSM8K_EXPLANATION_SPLIT_RE = re.compile(r"\b(?:because|since|as)\b", re.IGNORECASE)
_GSM8K_ABS_TOLERANCE = Decimal("1e-6")


def format_triviaqa_prompt(question: str) -> str:
    return TRIVIAQA_PROMPT_TEMPLATE.format(question=question)


def format_nq_open_prompt(question: str) -> str:
    return NQ_OPEN_PROMPT_TEMPLATE.format(question=question)


def format_web_questions_prompt(question: str) -> str:
    return WEBQUESTIONS_PROMPT_TEMPLATE.format(question=question)


def format_gsm8k_prompt(question: str) -> str:
    return GSM8K_PROMPT_TEMPLATE.format(question=question)


def parse_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(alias) for alias in value if str(alias).strip()]
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]

    if isinstance(parsed, list):
        return [str(alias) for alias in parsed if str(alias).strip()]
    return [str(parsed)] if str(parsed).strip() else []


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def clean_prediction(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text).strip()
    text = re.sub(r"(?is)^answer\s*:\s*", "", text).strip()
    text = text.splitlines()[0].strip()
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    return text.strip(" \t\r\n\"'`*_")


def clean_gsm8k_prediction(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = text.replace("\u2212", "-")
    text = re.sub(r"<<[^>=]*=([^>]*)>>", r" \1 ", text)
    text = re.sub(r"[*_`]", " ", text)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _decimal_from_number_text(text: str) -> Decimal | None:
    value = str(text).strip().replace(",", "").replace("$", "")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    if not value:
        return None

    try:
        with localcontext() as ctx:
            ctx.prec = 40
            if "/" in value:
                numerator, denominator = value.split("/", maxsplit=1)
                denom = Decimal(denominator)
                if denom == 0:
                    return None
                return Decimal(numerator) / denom
            return Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _format_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".")


def _gsm8k_number_candidates(text: str) -> list[tuple[str, Decimal]]:
    candidates = []
    for match in _GSM8K_NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        value = _decimal_from_number_text(raw)
        if value is not None:
            candidates.append((raw, value))
    return candidates


def extract_gsm8k_numeric_answer(text: str) -> str | None:
    cleaned = clean_gsm8k_prediction(text)
    if not cleaned:
        return None

    hash_matches = list(_GSM8K_HASH_RE.finditer(cleaned))
    if hash_matches:
        return hash_matches[-1].group("number").strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        line_match = _GSM8K_LINE_ANSWER_RE.match(line)
        if not line_match:
            continue
        body = _GSM8K_EXPLANATION_SPLIT_RE.split(line_match.group("body"), maxsplit=1)[0]
        candidates = _gsm8k_number_candidates(body)
        if candidates:
            return candidates[-1][0]

    direct_matches = list(_GSM8K_DIRECT_ANSWER_RE.finditer(cleaned))
    if direct_matches:
        return direct_matches[-1].group("number").strip()

    for line in reversed(lines):
        if "=" not in line:
            continue
        rhs = line.rsplit("=", maxsplit=1)[-1]
        candidates = _gsm8k_number_candidates(rhs)
        if candidates:
            return candidates[-1][0]

    candidates = _gsm8k_number_candidates(cleaned)
    return candidates[-1][0] if candidates else None


def extract_gsm8k_gold_answer(row: dict[str, Any]) -> str | None:
    for key in ("answer_value", "answer_normalized"):
        value = row.get(key)
        if value is None or not str(value).strip():
            continue
        candidate = extract_gsm8k_numeric_answer(str(value))
        if candidate is not None:
            return candidate

    for key in ("normalized_aliases", "aliases"):
        for alias in parse_aliases(row.get(key)):
            candidate = extract_gsm8k_numeric_answer(alias)
            if candidate is not None:
                return candidate

    reference_solution = row.get("reference_solution")
    if reference_solution:
        return extract_gsm8k_numeric_answer(str(reference_solution))
    return None


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    overlap = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in common)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_sequence_contains(container: str, contained: str) -> bool:
    container_tokens = normalize_answer(container).split()
    contained_tokens = normalize_answer(contained).split()
    if not container_tokens or not contained_tokens:
        return False
    if len(contained_tokens) > len(container_tokens):
        return False
    return any(
        container_tokens[start : start + len(contained_tokens)] == contained_tokens
        for start in range(0, len(container_tokens) - len(contained_tokens) + 1)
    )


def evaluate_triviaqa_answer(
    prediction: str,
    row: dict[str, Any],
    *,
    f1_threshold: float = 0.8,
) -> dict[str, Any]:
    cleaned_prediction = clean_prediction(prediction)
    normalized_prediction = normalize_answer(cleaned_prediction)

    gold_answers = parse_aliases(row.get("normalized_aliases"))
    if row.get("answer_normalized"):
        gold_answers.append(str(row["answer_normalized"]))
    if row.get("answer_value"):
        gold_answers.append(normalize_answer(row["answer_value"]))
    gold_answers = sorted(
        {normalize_answer(answer) for answer in gold_answers if normalize_answer(answer)}
    )

    exact_match = normalized_prediction in gold_answers
    scored_aliases = [(alias, token_f1(normalized_prediction, alias)) for alias in gold_answers]
    matched_alias, best_f1 = max(scored_aliases, key=lambda item: item[1], default=("", 0.0))

    return {
        "prediction_raw": prediction,
        "prediction_clean": cleaned_prediction,
        "prediction_normalized": normalized_prediction,
        "exact_match": exact_match,
        "f1": best_f1,
        "matched_alias": matched_alias,
        "is_correct": bool(exact_match or best_f1 >= f1_threshold),
    }


def evaluate_nq_open_answer(
    prediction: str,
    row: dict[str, Any],
    *,
    f1_threshold: float = 0.8,
) -> dict[str, Any]:
    cleaned_prediction = clean_prediction(prediction)
    normalized_prediction = normalize_answer(cleaned_prediction)

    gold_answers = parse_aliases(row.get("normalized_aliases"))
    if row.get("answer_normalized"):
        gold_answers.append(str(row["answer_normalized"]))
    if row.get("answer_value"):
        gold_answers.append(normalize_answer(row["answer_value"]))
    gold_answers = sorted(
        {normalize_answer(answer) for answer in gold_answers if normalize_answer(answer)}
    )

    exact_match = normalized_prediction in gold_answers
    scored_aliases = [(alias, token_f1(normalized_prediction, alias)) for alias in gold_answers]
    matched_alias, best_f1 = max(scored_aliases, key=lambda item: item[1], default=("", 0.0))

    return {
        "prediction_raw": prediction,
        "prediction_clean": cleaned_prediction,
        "prediction_normalized": normalized_prediction,
        "exact_match": exact_match,
        "f1": best_f1,
        "matched_alias": matched_alias,
        "is_correct": bool(exact_match or best_f1 >= f1_threshold),
    }


def evaluate_web_questions_answer(
    prediction: str,
    row: dict[str, Any],
    *,
    f1_threshold: float = 0.8,
) -> dict[str, Any]:
    cleaned_prediction = clean_prediction(prediction)
    normalized_prediction = normalize_answer(cleaned_prediction)

    gold_answers = parse_aliases(row.get("answers"))
    gold_answers.extend(parse_aliases(row.get("normalized_aliases")))
    if row.get("answer_normalized"):
        gold_answers.append(str(row["answer_normalized"]))
    if row.get("answer_value"):
        gold_answers.append(normalize_answer(row["answer_value"]))
    gold_answers = sorted(
        {normalize_answer(answer) for answer in gold_answers if normalize_answer(answer)}
    )

    exact_match = normalized_prediction in gold_answers
    containment_match = any(
        token_sequence_contains(normalized_prediction, alias)
        or token_sequence_contains(alias, normalized_prediction)
        for alias in gold_answers
    )
    scored_aliases = [(alias, token_f1(normalized_prediction, alias)) for alias in gold_answers]
    matched_alias, best_f1 = max(scored_aliases, key=lambda item: item[1], default=("", 0.0))

    return {
        "prediction_raw": prediction,
        "prediction_clean": cleaned_prediction,
        "prediction_normalized": normalized_prediction,
        "exact_match": exact_match,
        "containment_match": containment_match,
        "f1": best_f1,
        "matched_alias": matched_alias,
        "is_correct": bool(exact_match or containment_match or best_f1 >= f1_threshold),
    }


def evaluate_gsm8k_answer(
    prediction: str,
    row: dict[str, Any],
    *,
    abs_tolerance: Decimal = _GSM8K_ABS_TOLERANCE,
) -> dict[str, Any]:
    cleaned_prediction = clean_gsm8k_prediction(prediction)
    prediction_answer = extract_gsm8k_numeric_answer(cleaned_prediction)
    gold_answer = extract_gsm8k_gold_answer(row)

    prediction_value = _decimal_from_number_text(prediction_answer or "")
    gold_value = _decimal_from_number_text(gold_answer or "")
    abs_error = (
        abs(prediction_value - gold_value)
        if prediction_value is not None and gold_value is not None
        else None
    )
    numeric_match = bool(abs_error is not None and abs_error <= abs_tolerance)

    return {
        "prediction_raw": prediction,
        "prediction_clean": prediction_answer or "",
        "prediction_text_clean": cleaned_prediction,
        "prediction_normalized": _format_decimal(prediction_value),
        "gold_answer": gold_answer or "",
        "gold_normalized": _format_decimal(gold_value),
        "numeric_match": numeric_match,
        "abs_error": float(abs_error) if abs_error is not None else None,
        "is_correct": numeric_match,
    }
