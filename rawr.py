"""
Uses LLaMA3 (via Ollama) to generate a natural-language story for each lead,
grounded in SHAP values, topic alignment signals, and lead context.
"""
import json
import time
import requests
import pandas as pd
from tqdm import tqdm

# CONFIG
OLLAMA_URL        = "http://localhost:11434/api/generate"
MODEL             = "llama3"
INPUT_FILE        = "synthetic_leads_shap - copy.xlsx"
OUTPUT_FILE       = "leads_with_explanations_v0.xlsx"
TOP_N_FEATURES    = 5    # how many SHAP drivers to highlight in the prompt
MAX_TOKENS        = 200    # cap on LLM response length
TEMPERATURE       = 0.4    # lower = more consistent/factual tone
BATCH_SIZE        = 10     # rows to process before saving a checkpoint
REQUEST_TIMEOUT   = 300    # seconds before giving up on one LLM call

# STEP 1 — LOAD & PARSE
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Loaded {len(df)} leads with {len(df.columns)} columns.")
    return df

def extract_shap_features(row: pd.Series, top_n: int = TOP_N_FEATURES) -> dict:
    """
    Pull all (feature_name, shap_value) pairs from a row, sort by |SHAP|,
    and return the top N positive drivers and top N negative drivers.
    """
    pairs = []
    i = 1
    while f"feature_name{i:02d}" in row.index:
        fname = row[f"feature_name{i:02d}"]
        fval  = row[f"shap_value{i:02d}"]
        if pd.notna(fname) and pd.notna(fval):
            pairs.append((fname, float(fval)))
        i += 1

    pairs.sort(key=lambda x: x[1], reverse=True)
    positive_drivers = [(f, v) for f, v in pairs if v > 0][:top_n]
    negative_drivers = [(f, v) for f, v in pairs if v < 0][-top_n:][::-1]  # most negative first

    return {
        "all_pairs":        pairs,
        "positive_drivers": positive_drivers,
        "negative_drivers": negative_drivers,
    }

def extract_topic_signals(row: pd.Series) -> list[dict]:
    """
    Parse topic alignment columns into a clean list of active/inactive signals.
    """
    topics = [
        ("Lead Description Completeness",    "Lead_Description_Completeness"),
        ("Lead Score & Stage",               "Lead_Score_and_Stage"),
        ("Corporate Email Address",          "Corporate_Email_Address"),
        ("Existing Customer Validity",       "Existing_Customer_Account_Validity"),
        ("Source Awareness",                 "Source_Awareness"),
        ("Early Intent Signal",              "Early_Intent"),
        ("Strong Commercial Intent",         "Strong_Commercial_Intent"),
        ("FSE Generated Lead",               "FSE_Generated"),
    ]
    signals = []
    for label, col in topics:
        alignment_col = f"{col}_alignment"
        shap_col      = f"{col}_shap"
        if alignment_col in row.index:
            signals.append({
                "label":     label,
                "aligned":   bool(row[alignment_col]),
                "shap":      float(row[shap_col]) if shap_col in row.index else 0.0,
            })
    return signals
# STEP 2 — PROMPT ENGINEERING

SYSTEM_PROMPT = """
You are an expert sales analyst.

Your task is to generate:
1. A natural-language lead narrative for a sales representative.
2. A confidence score showing how reliable the narrative is based on:
   - completeness of lead data,
   - consistency of SHAP signals,
   - engagement evidence,
   - missing or conflicting information.

OUTPUT RULES:
- Write ONLY valid JSON.
- Do not include markdown.
- Do not include explanations outside JSON.

Required JSON format:

{
  "lead_story": "4-6 sentence narrative paragraph",
  "confidence_pct": 85,
  "confidence_reason": "Short explanation of why confidence is high or low"
}

WRITING RULES:
- Use plain business English.
- No bullet points.
- No headers.
- Never invent information.
- Mention strengths and risks naturally.
- End with a recommended sales action.
- Interpret signals instead of repeating raw SHAP numbers.
"""

EXAMPLE_OUTPUTS = """
Example 1:
{
  "lead_story": "This lead shows strong conversion potential due to high engagement activity, clear product interest, and a verified corporate account. The opportunity is supported by strong commercial intent signals and healthy demographic alignment, suggesting the prospect is actively evaluating solutions. However, limited phone availability and incomplete lead descriptions may slow direct outreach effectiveness. The lead is currently rated as high quality with a strong predicted conversion likelihood, making it suitable for immediate follow-up by the regional sales team. A personalized outreach focused on product fit and timing would likely improve conversion chances.",
  "confidence_pct": 91,
  "confidence_reason": "High confidence due to strong engagement signals, verified corporate data, multiple aligned intent indicators, and minimal missing information."
}

Example 2:
{
  "lead_story": "This lead demonstrates moderate sales potential but contains several data gaps that reduce certainty around purchase intent. While the account appears to have some engagement activity and aligns with early intent indicators, weak contact coverage and limited supporting behavioral evidence create uncertainty around readiness to convert. The predicted conversion likelihood remains moderate, though negative conversion drivers suggest the lead may still be in an exploratory phase. Sales outreach should focus on validating current business needs and confirming decision-maker engagement before prioritizing further resources.",
  "confidence_pct": 63,
  "confidence_reason": "Moderate confidence because several important engagement and contact signals are missing or weak."
}
"""

def build_prompt(row: pd.Series, shap_info: dict, topic_signals: list[dict]) -> str:

    lead_block = f"""
LEAD SUMMARY
------------
Lead ID       : {row['Lead_ID']}
Company       : {row['Company_Account']}
Lead Status   : {row['Lead_Status']}
Lead Source   : {row['Lead_Source']}
Region        : {row['Lead_Region']} / {row['Sub_Region']}
Market        : {row['Account_Market']}
Account Type  : {row['Account_Type']}
Product       : {row['Product_Interest']}
Aircraft      : {row['Aircraft_Type']}
Est. Value    : ${row['Estimated_Value_USD']:,.0f}
Loyalty       : {row['Loyalty_Segment'].capitalize()}
Account Exists: {row['Account_Exists']}
Phone         : {'Available' if row['Phone_Available'] else 'Not Available'}
Corporate Email: {'Yes' if row['Corporate_Email'] else 'No'}
""".strip()

    score_block = f"""
MODEL SCORES
------------
Predicted Conversion Probability : {row['pred_proba']:.1%}
Lead Rating                      : {row['lead_rating'].upper()}
AQL Category                     : {row['aql_category'].replace('_', ' ').title()}
Behavior Score                   : {row['Behavior_Score']:.1f}/100
Demographic Score                : {row['Demographic_Score']:.1f}/100
Engagement Score                 : {row['Engagement_Score']:.1f}/100
Model Confidence                 : {row['pred_confidence']:.1%}
""".strip()

    pos_lines = "\n".join(
        f"(+) {f}" for f, _ in shap_info["positive_drivers"]
    )

    neg_lines = "\n".join(
        f"(-) {f}" for f, _ in shap_info["negative_drivers"]
    )

    shap_block = f"""
KEY CONVERSION DRIVERS
----------------------
Positive Signals:
{pos_lines if pos_lines else 'None'}

Negative Signals:
{neg_lines if neg_lines else 'None'}
""".strip()

    aligned     = [s["label"] for s in topic_signals if s["aligned"]]
    not_aligned = [s["label"] for s in topic_signals if not s["aligned"]]

    topic_block = f"""
TOPIC ALIGNMENT SIGNALS
-----------------------
Aligned:
{', '.join(aligned) if aligned else 'None'}

Not Aligned:
{', '.join(not_aligned) if not_aligned else 'None'}
""".strip()

    return f"""
{EXAMPLE_OUTPUTS}

Now generate the JSON output for this lead.

{lead_block}

{score_block}

{shap_block}

{topic_block}
""".strip()

# STEP 3 — LLM CALL
def call_ollama(system: str, user_prompt: str) -> str:
    """
    Send a prompt to the local Ollama LLaMA3 instance and return the response text.
    """
    payload = {
        "model":  MODEL,
        "prompt": f"<|system|>\n{system}\n<|user|>\n{user_prompt}\n<|assistant|>",
        "stream": False,
        "options": {
            "temperature":   TEMPERATURE,
            "num_predict":   MAX_TOKENS,
            "stop":          ["<|user|>", "<|system|>"],
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Is 'ollama serve' running?"
    except requests.exceptions.Timeout:
        return "[ERROR] Request timed out. Try reducing MAX_TOKENS or simplifying the prompt."
    except Exception as e:
        return f"[ERROR] {str(e)}"

# STEP 4 — PROCESS ALL LEADS
def process_leads(df: pd.DataFrame) -> pd.DataFrame:

    explanations       = []
    prompts_used       = []
    durations          = []
    llm_confidences    = []
    llm_conf_reasons   = []

    print(f"\nRunning LLaMA3 explanations for {len(df)} leads...\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):

        shap_info     = extract_shap_features(row)
        topic_signals = extract_topic_signals(row)

        prompt = build_prompt(row, shap_info, topic_signals)

        t0 = time.time()

        response = call_ollama(SYSTEM_PROMPT, prompt)

        elapsed = round(time.time() - t0, 2)

        # defaults
        lead_story       = response
        confidence_pct   = None
        confidence_reason = None

        try:
            parsed = json.loads(response)

            lead_story        = parsed.get("lead_story", "")
            confidence_pct    = parsed.get("confidence_pct", None)
            confidence_reason = parsed.get("confidence_reason", "")

        except Exception:
            pass

        explanations.append(lead_story)
        prompts_used.append(prompt)
        durations.append(elapsed)
        llm_confidences.append(confidence_pct)
        llm_conf_reasons.append(confidence_reason)

        if (idx + 1) % BATCH_SIZE == 0:
            _checkpoint_save(
                df,
                explanations,
                prompts_used,
                durations,
                llm_confidences,
                llm_conf_reasons,
                idx
            )

    df["llm_explanation"]       = explanations
    df["prompt_used"]           = prompts_used
    df["llm_duration_sec"]      = durations
    df["llm_confidence_pct"]    = llm_confidences
    df["llm_confidence_reason"] = llm_conf_reasons

    return df

def _checkpoint_save(
    df,
    explanations,
    prompts_used,
    durations,
    llm_confidences,
    llm_conf_reasons,
    up_to_idx
):

    temp = df.iloc[:len(explanations)].copy()

    temp["llm_explanation"]       = explanations
    temp["prompt_used"]           = prompts_used
    temp["llm_duration_sec"]      = durations
    temp["llm_confidence_pct"]    = llm_confidences
    temp["llm_confidence_reason"] = llm_conf_reasons

    ckpt = OUTPUT_FILE.replace(".xlsx", f"_checkpoint_{up_to_idx+1}.xlsx")

    temp.to_excel(ckpt, index=False)

    tqdm.write(f"✓ Checkpoint saved → {ckpt}")

# STEP 5 — SAVE OUTPUT
def save_output(df: pd.DataFrame, path: str):
    from openpyxl.styles import Font, PatternFill, Alignment, PatternFill
    from openpyxl import load_workbook
    df.to_excel(path, index=False, engine="openpyxl")
    wb = load_workbook(path)
    ws = wb.active

    # Style header row
    header_fill = PatternFill("solid", start_color="1F4E79")
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    for cell in ws[1]:
        cell.fill   = header_fill
        cell.font   = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    # Auto-width + wrap explanation column
    for col in ws.columns:
        col_letter = col[0].column_letter
        header_val = ws[f"{col_letter}1"].value or ""
        if "explanation" in str(header_val).lower():
            ws.column_dimensions[col_letter].width = 80
            for cell in col[1:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        elif "prompt" in str(header_val).lower():
            ws.column_dimensions[col_letter].width = 60
        else:
            ws.column_dimensions[col_letter].width = 18

    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30
    wb.save(path)
    print(f"\nFinal output saved → {path}")

if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    final_df = process_leads(df)
    save_output(final_df, OUTPUT_FILE)
    print("Done.")