#!/usr/bin/env python3
import os
import re

import pandas as pd
import requests

INPUT_FILE = "shap_values.xlsx"
OUTPUT_FILE = "shap_values_explained.xlsx"
EXPLANATION_COLUMN = "explanation"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
VALUE_PREFIX = "shap_value"
NAME_PREFIX = "feature_name"
TOP_N = 5
TEMPERATURE = 0.2
MAX_TOKENS = 300
STOP_TOKENS = ["<|user|>", "<|system|>"]
TIMEOUT = 60

DEFAULT_EXCLUDE = {
    "id",
    "customer_id",
    "lead_id",
    "target",
    "label",
    "result",
    "model_score",
    "base_value",
}

DEFAULT_PROMPT = (
    "You are a business analyst writing for non-technical stakeholders.\n"
    "Given contribution values for one row, write a short, clear paragraph (3-5 sentences) "
    "explaining what most influenced the score in plain language.\n"
    "Use only the data provided. Include key contribution values (with + or -), and reference any "
    "context metrics to improve transparency and traceability. Avoid jargon.\n"
    "Use the contribution metrics to summarize the overall tilt (net impact, total positives vs "
    "negatives), and call out a dominant factor if one clearly stands out.\n"
    "Context metrics (numeric):\n"
    "{context_metrics}\n"
    "Context fields (other):\n"
    "{context_fields}\n"
    "Contribution metrics:\n"
    "{contribution_metrics}\n"
    "Top contributors:\n"
    "{top_contributors}\n"
    "Top positive contributors:\n"
    "{top_positive}\n"
    "Top negative contributors:\n"
    "{top_negative}\n"
    "All contributors:\n"
    "{all_contributors}\n"
)

SYSTEM_PROMPT = (
    "Write one concise paragraph in plain business English. "
    "Ground every statement in the provided data, avoid jargon, and do not invent details. "
    "Mention key numeric metrics and the most important drivers."
)


def detect_indexed_pairs(df, value_prefix, name_prefix):
    pairs = []

    if value_prefix and name_prefix:
        value_pattern = re.compile(
            rf"^{re.escape(value_prefix)}(\d+)$", re.IGNORECASE
        )
        name_pattern = re.compile(
            rf"^{re.escape(name_prefix)}(\d+)$", re.IGNORECASE
        )
        value_map = {}
        name_map = {}

        for col in df.columns:
            col_str = str(col)
            value_match = value_pattern.match(col_str)
            if value_match:
                value_map[int(value_match.group(1))] = col
                continue
            name_match = name_pattern.match(col_str)
            if name_match:
                name_map[int(name_match.group(1))] = col

        for idx in sorted(set(value_map) & set(name_map)):
            pairs.append((value_map[idx], name_map[idx]))

        if pairs:
            return pairs

    index_pattern = re.compile(r"^(.*?)(\d+)$")
    numeric_candidates = {}
    text_candidates = {}

    for col in df.columns:
        col_str = str(col)
        match = index_pattern.match(col_str)
        if not match:
            continue

        idx = int(match.group(2))
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.setdefault(idx, []).append(col)
        else:
            text_candidates.setdefault(idx, []).append(col)

    for idx in sorted(set(numeric_candidates) & set(text_candidates)):
        pairs.append((numeric_candidates[idx][0], text_candidates[idx][0]))

    return pairs


def infer_numeric_columns(df, include, exclude):
    if include:
        missing = [col for col in include if col not in df.columns]
        if missing:
            raise ValueError(f"Missing numeric columns: {', '.join(missing)}")
        return include

    exclude_set = {item.casefold() for item in exclude}
    numeric_cols = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    return [
        col for col in numeric_cols if str(col).casefold() not in exclude_set
    ]


def extract_contributions(row, pairs, numeric_cols):
    contributions = []
    if pairs:
        for value_col, name_col in pairs:
            name = row.get(name_col)
            value = row.get(value_col)

            if pd.isna(name) or pd.isna(value):
                continue

            name_text = str(name).strip()
            if not name_text:
                continue

            try:
                value_num = float(value)
            except (TypeError, ValueError):
                continue

            contributions.append((name_text, value_num))
    else:
        for col in numeric_cols:
            value = row.get(col)
            if pd.isna(value):
                continue

            try:
                value_num = float(value)
            except (TypeError, ValueError):
                continue

            contributions.append((str(col), value_num))

    return contributions


def compute_contribution_metrics(contributions):
    if not contributions:
        return {
            "total_positive": 0.0,
            "total_negative": 0.0,
            "net_impact": 0.0,
            "total_abs": 0.0,
            "pos_count": 0,
            "neg_count": 0,
            "dominant_feature": "none",
            "dominant_value": 0.0,
            "dominance_ratio": 0.0,
        }

    total_positive = sum(value for _, value in contributions if value > 0)
    total_negative = sum(value for _, value in contributions if value < 0)
    total_abs = sum(abs(value) for _, value in contributions)
    net_impact = total_positive + total_negative
    pos_count = sum(1 for _, value in contributions if value > 0)
    neg_count = sum(1 for _, value in contributions if value < 0)
    dominant_feature, dominant_value = max(
        contributions, key=lambda item: abs(item[1])
    )
    dominance_ratio = abs(dominant_value) / total_abs if total_abs else 0.0

    return {
        "total_positive": total_positive,
        "total_negative": total_negative,
        "net_impact": net_impact,
        "total_abs": total_abs,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "dominant_feature": dominant_feature,
        "dominant_value": dominant_value,
        "dominance_ratio": dominance_ratio,
    }


def format_lines(label_value_pairs):
    if not label_value_pairs:
        return "- none"
    return "\n".join(f"- {label}: {value}" for label, value in label_value_pairs)


def format_contributors(contributions):
    if not contributions:
        return "- none"
    return "\n".join(
        f"- {name}: {value:+.4f}" for name, value in contributions
    )


def build_prompt(
    contributions,
    top_n,
    template,
    context_metrics,
    context_fields,
):
    if not contributions:
        return template.format(
            context_metrics=format_lines(context_metrics),
            context_fields=format_lines(context_fields),
            contribution_metrics="- none",
            top_contributors="- none",
            top_positive="- none",
            top_negative="- none",
            all_contributors="- none",
            top_n=top_n,
        )

    top = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:top_n]
    pos = [(name, val) for name, val in top if val > 0]
    neg = [(name, val) for name, val in top if val < 0]
    all_sorted = sorted(
        contributions, key=lambda item: abs(item[1]), reverse=True
    )

    metrics = compute_contribution_metrics(contributions)
    metric_lines = [
        ("total_positive", f"{metrics['total_positive']:+.4f}"),
        ("total_negative", f"{metrics['total_negative']:+.4f}"),
        ("net_impact", f"{metrics['net_impact']:+.4f}"),
        ("total_abs", f"{metrics['total_abs']:.4f}"),
        ("pos_count", str(metrics["pos_count"])),
        ("neg_count", str(metrics["neg_count"])),
        (
            "dominant_feature",
            f"{metrics['dominant_feature']} ({metrics['dominant_value']:+.4f})",
        ),
        (
            "dominance_ratio",
            f"{metrics['dominance_ratio'] * 100:.1f}%",
        ),
    ]

    return template.format(
        context_metrics=format_lines(context_metrics),
        context_fields=format_lines(context_fields),
        contribution_metrics=format_lines(metric_lines),
        top_contributors=format_contributors(top),
        top_positive=format_contributors(pos),
        top_negative=format_contributors(neg),
        all_contributors=format_contributors(all_sorted),
        top_n=top_n,
    )


def call_ollama(url, model, prompt, system, temperature, timeout, max_tokens, stop):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens:
        options["num_predict"] = max_tokens
    if stop:
        options["stop"] = stop
    if options:
        payload["options"] = options

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Failed to call Ollama at {url}: {exc}") from exc

    try:
        parsed = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid JSON from Ollama: {response.text[:200]}"
        ) from exc

    if "response" not in parsed:
        raise RuntimeError(f"Unexpected Ollama response: {parsed}")

    return parsed["response"].strip()


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    pairs = detect_indexed_pairs(df, VALUE_PREFIX.strip(), NAME_PREFIX.strip())
    numeric_cols = []
    used_cols = set()
    if not pairs:
        numeric_cols = infer_numeric_columns(df, [], DEFAULT_EXCLUDE)
        if not numeric_cols:
            raise SystemExit("No paired columns or numeric columns found to explain.")
        used_cols.update(numeric_cols)
    else:
        used_cols.update([value_col for value_col, _ in pairs])
        used_cols.update([name_col for _, name_col in pairs])

    context_metric_cols = [
        col
        for col in df.columns
        if col not in used_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    context_field_cols = [
        col
        for col in df.columns
        if col not in used_cols and col not in context_metric_cols
    ]

    explanations = []
    for _, row in df.iterrows():
        contributions = extract_contributions(row, pairs, numeric_cols)
        context_metrics = []
        for col in context_metric_cols:
            value = row.get(col)
            if pd.isna(value):
                continue
            context_metrics.append((str(col), f"{float(value):.4f}"))

        context_fields = []
        for col in context_field_cols:
            value = row.get(col)
            if pd.isna(value):
                continue
            context_fields.append((str(col), str(value)))

        prompt = build_prompt(
            contributions,
            TOP_N,
            DEFAULT_PROMPT,
            context_metrics,
            context_fields,
        )
        try:
            explanation = call_ollama(
                OLLAMA_URL,
                MODEL_NAME,
                prompt,
                system=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                timeout=TIMEOUT,
                max_tokens=MAX_TOKENS,
                stop=STOP_TOKENS,
            )
        except RuntimeError as exc:
            explanation = f"[ERROR] {exc}"
        explanations.append(explanation)

    df_out = df.copy()
    df_out[EXPLANATION_COLUMN] = explanations
    df_out.to_excel(OUTPUT_FILE, index=False)

    print(
        f"Wrote {OUTPUT_FILE} with '{EXPLANATION_COLUMN}' for {len(df_out)} rows."
    )


if __name__ == "__main__":
    main()
