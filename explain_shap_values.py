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
    "Given SHAP contributions for one row, write a short, clear paragraph (3-5 sentences) "
    "explaining what most influenced the score in plain language.\n"
    "Use only the features provided. Say which factors pushed the score up or down, and "
    "avoid jargon.\n"
    "Top contributors:\n"
    "{top_contributors}\n"
    "Top positive contributors:\n"
    "{top_positive}\n"
    "Top negative contributors:\n"
    "{top_negative}\n"
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


def build_prompt(contributions, top_n, template):
    if not contributions:
        top_lines = "- none"
        pos_lines = "- none"
        neg_lines = "- none"
        return template.format(
            top_contributors=top_lines,
            top_positive=pos_lines,
            top_negative=neg_lines,
            top_n=top_n,
        )

    top = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:top_n]
    pos = [(name, val) for name, val in top if val > 0]
    neg = [(name, val) for name, val in top if val < 0]

    top_lines = "\n".join(f"- {name}: {val:+.4f}" for name, val in top)
    pos_lines = (
        "\n".join(f"- {name}: {val:+.4f}" for name, val in pos)
        if pos
        else "- none"
    )
    neg_lines = (
        "\n".join(f"- {name}: {val:+.4f}" for name, val in neg)
        if neg
        else "- none"
    )

    return template.format(
        top_contributors=top_lines,
        top_positive=pos_lines,
        top_negative=neg_lines,
        top_n=top_n,
    )


def call_ollama(url, model, prompt, system, temperature, timeout):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

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
    if not pairs:
        numeric_cols = infer_numeric_columns(df, [], DEFAULT_EXCLUDE)
        if not numeric_cols:
            raise SystemExit("No paired columns or numeric columns found to explain.")

    explanations = []
    for _, row in df.iterrows():
        contributions = extract_contributions(row, pairs, numeric_cols)
        prompt = build_prompt(contributions, TOP_N, DEFAULT_PROMPT)
        explanation = call_ollama(
            OLLAMA_URL,
            MODEL_NAME,
            prompt,
            system="",
            temperature=TEMPERATURE,
            timeout=TIMEOUT,
        )
        explanations.append(explanation)

    df_out = df.copy()
    df_out[EXPLANATION_COLUMN] = explanations
    df_out.to_excel(OUTPUT_FILE, index=False)

    print(
        f"Wrote {OUTPUT_FILE} with '{EXPLANATION_COLUMN}' for {len(df_out)} rows."
    )


if __name__ == "__main__":
    main()
