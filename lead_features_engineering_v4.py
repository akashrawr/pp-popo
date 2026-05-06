import pandas as pd
import json
import requests
import time

# ======================================
# CONFIG
# ======================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME  = "llama3"
LEADS_FILE  = "leads_input.xlsx"          # ← your Excel input file
MAX_LEADS   = 20                          # ← set to None to process all rows

ALLOWED_VALUES = {
    "intent_level":            {"Low", "Medium", "High"},
    "urgency":                 {"Urgent", "Moderate", "Not Given"},
    "budget_signal":           {"High Budget", "Budget Constrained", "Not Given"},
    "lead_scale":              {"Enterprise", "Mid-Market", "SMB", "Startup", "Unknown"},
    "intuition_score":         {"Hot", "Lukewarm", "Cold"},
    "decision_maker_involved": {"Yes", "No", "Unknown"},
    "product_match_strength":  {"Strong", "Partial", "Weak"},
    "type_of_customer":        {"Willing to Buy", "Researching", "Other"},
    "qualification_status":    {"Qualified", "Needs Follow-Up", "Not Qualified"},
    "confidence_level":        {"High", "Medium", "Low"},
    "lead_quality_score":      {"Strong", "Moderate", "Weak"},
    "product_intent_detected": {"Yes", "No", "Unclear"},
    "estimated_deal_value":    {"High", "Medium", "Low", "Unknown"},
}


# ======================================
# STEP 1: LOAD FROM EXCEL
# Each column read separately, then
# zipped together row by row per lead
# ======================================

def check_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["customer_comments", "product_interest", "product_info", "lead_description"]
    df["empty_fields"] = df[cols].apply(
        lambda row: sum(1 for v in row if pd.isna(v) or str(v).strip() == ""), axis=1
    )
    sparse = df[df["empty_fields"] >= 3]
    if not sparse.empty:
        print(f"  [WARN] {len(sparse)} leads have 3+ empty fields — results may be unreliable: rows {sparse['lead_id'].tolist()}")
    else:
        print(f"  [OK] All leads passed data quality check")
    return df


def load_leads_from_excel(filepath: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, engine="openpyxl")

    # Read each column separately as its own list
    customer_comments = raw["customer_comments"].tolist()
    product_interest  = raw["product_interest"].tolist()
    product_info      = raw["product_info"].tolist()
    lead_description  = raw["lead_description"].tolist()

    # Zip together row by row — one coherent lead per row
    rows = []
    for i, (cc, pi, pinfo, ld) in enumerate(
        zip(customer_comments, product_interest, product_info, lead_description),
        start=1
    ):
        rows.append({
            "lead_id":           i,
            "customer_comments": cc,
            "product_interest":  pi,
            "product_info":      pinfo,
            "lead_description":  ld,
        })

    df = pd.DataFrame(rows)

    # Limit to MAX_LEADS rows if set
    if MAX_LEADS is not None:
        df = df.head(MAX_LEADS)

    # Run data quality check
    df = check_data_quality(df)

    return df


# ======================================
# STEP 2: BUILD UNIFIED LEAD CONTEXT
# ======================================

def build_unified_context(row):
    return f"""
You are analyzing ONE lead. The four fields below all describe the SAME lead.
Read ALL four fields together — clues in one field may clarify or contradict another.
Do not analyse any field in isolation.
The product_interest field is the most important signal for what the lead wants to buy.

--- CUSTOMER COMMENTS ---
{row['customer_comments']}

--- PRODUCT INTEREST ---
{row['product_interest']}

--- PRODUCT INFO (product the lead is being shown) ---
{row['product_info']}

--- LEAD DESCRIPTION ---
{row['lead_description']}
""".strip()


# ======================================
# STEP 3: SYSTEM PROMPT
# ======================================

SYSTEM_PROMPT = """
You are a B2B lead qualification assistant performing feature engineering on sales lead data.
This is part of a larger pipeline — leads have already been pre-filtered. Focus only on extracting
structured signals from the text. Do not flag missing contact details or basic company info.

Read ALL four fields as ONE lead context. Cross-reference them.
The product_interest field is an important signal, but must be validated against other fields.

Return STRICT JSON ONLY. No explanation. No markdown fences. No text outside the JSON.

### EXAMPLES

EXAMPLE 1:
INPUT:
--- CUSTOMER COMMENTS ---
We need a CRM that integrates with our ERP system. Our sales team is struggling with manual data entry.
--- PRODUCT INTEREST ---
CRM with ERP integration
--- PRODUCT INFO ---
Cloud CRM platform with automation and reporting
--- LEAD DESCRIPTION ---
Mid-sized company looking to improve sales efficiency and reduce manual work.
OUTPUT:
{
  "extracted_product_category": "CRM",
  "extracted_product_features": {
    "product_interest": ["ERP integration", "automation", "sales reporting"],
    "support_interest": []
  },
  "intents": ["evaluate CRM vendors", "improve sales efficiency"],
  "intent_level": "High",
  "purchase_intent_reason": "Manual data entry inefficiencies are slowing down the sales team and require immediate improvement.",
  "urgency": "Moderate",
  "budget_signal": "Not Given",
  "lead_scale": "Mid-Market",
  "intuition_score": "Hot",
  "decision_maker_involved": "Unknown",
  "pain_points": ["manual data entry", "inefficient sales processes"],
  "type_of_customer": "Willing to Buy",
  "product_match_strength": "Strong",
  "lead_quality_score": "Strong",
  "qualification_status": "Qualified",
  "confidence_level": "High",
  "reasoning": "Strong intent with clear pain points and good product fit indicates a qualified lead.",
  "reasoning_evidence": {
    "intent_level_evidence": "need a CRM that integrates with ERP system",
    "urgency_evidence": "struggling with manual data entry",
    "budget_evidence": "no budget mentioned",
    "intuition_evidence": "clear problem and defined solution",
    "pain_point_evidence": "manual data entry"
  },
  "product_intent_detected": "Yes",
  "products_identified": ["CRM platform", "ERP integration module"],
  "estimated_deal_value": "Medium",
  "estimated_deal_value_range": "$10,000-$50,000",

  "product_deal_evidence": {
  "product_intent_evidence": "we need a CRM that integrates with our ERP system",
  "products_identified_evidence": "CRM platform with ERP integration, automation and reporting",
  "deal_value_evidence": "mid-sized company, improving sales efficiency across the team",
  "deal_value_range_evidence": "mid-sized company with department-wide rollout on a mid-market package"
}
}

EXAMPLE 2:
INPUT:
--- CUSTOMER COMMENTS ---
Just exploring tools for analytics. No immediate plans.
--- PRODUCT INTEREST ---
Analytics dashboard
--- PRODUCT INFO ---
Business intelligence platform
--- LEAD DESCRIPTION ---
Small startup browsing options.
OUTPUT:
{
  "extracted_product_category": "Analytics",
  "extracted_product_features": {
    "product_interest": ["dashboard", "data visualization"],
    "support_interest": []
  },
  "intents": ["explore analytics tools"],
  "intent_level": "Low",
  "purchase_intent_reason": "The lead is casually exploring analytics tools without a defined business trigger.",
  "urgency": "Not Given",
  "budget_signal": "Not Given",
  "lead_scale": "Startup",
  "intuition_score": "Cold",
  "decision_maker_involved": "Unknown",
  "pain_points": ["lack of analytics tools"],
  "type_of_customer": "Researching",
  "product_match_strength": "Partial",
  "lead_quality_score": "Moderate",
  "qualification_status": "Needs Follow-Up",
  "confidence_level": "Medium",
  "reasoning": "Low intent and exploratory behavior indicate the lead is not ready to buy.",
  "reasoning_evidence": {
    "intent_level_evidence": "just exploring tools",
    "urgency_evidence": "no immediate plans",
    "budget_evidence": "not mentioned",
    "intuition_evidence": "browsing behavior without commitment",
    "pain_point_evidence": "exploring tools"
  },
  "product_intent_detected": "No",
  "products_identified": [],
  "estimated_deal_value": "Unknown",
  "estimated_deal_value_range": "Unknown",

  "product_deal_evidence": {
  "product_intent_evidence": "just exploring tools, no immediate plans",
  "products_identified_evidence": "analytics dashboard mentioned but no commitment",
  "deal_value_evidence": "small startup browsing, no budget mentioned",
  "deal_value_range_evidence": "startup with no budget signals and exploratory behaviour"
}
}

### NOW ANALYZE THIS LEAD

Use this EXACT structure:

{
  "extracted_product_category": "CRM / ERP Integration / Customer Support / Analytics / Automation / Other",

  "extracted_product_features": {
    "product_interest": ["specific product feature 1", "specific product feature 2"],
    "support_interest": ["support or service need 1", "support or service need 2"]
  },

  "intents": ["primary intent verb phrase", "secondary intent if present"],
  "intent_level": "Low / Medium / High",

  "purchase_intent_reason": "One sentence: what specific trigger or business event is driving them to buy now.",

  "urgency": "Urgent / Moderate / Not Given",
  "budget_signal": "High Budget / Budget Constrained / Not Given",

  "lead_scale": "Enterprise / Mid-Market / SMB / Startup / Unknown",

  "intuition_score": "Hot / Lukewarm / Cold",

  "decision_maker_involved": "Yes / No / Unknown",

  "pain_points": ["pain point 1", "pain point 2"],

  "type_of_customer": "Willing to Buy / Researching / Other",
  "product_match_strength": "Strong / Partial / Weak",

  "lead_quality_score": "Strong / Moderate / Weak",

  "qualification_status": "Qualified / Needs Follow-Up / Not Qualified",
  "confidence_level": "High / Medium / Low",

  "reasoning": "One sentence explaining the qualification decision.",

  "reasoning_evidence": {
    "intent_level_evidence": "Quote or paraphrase the exact words that set the intent level.",
    "urgency_evidence": "Quote or paraphrase the exact words that set urgency.",
    "budget_evidence": "Quote or paraphrase the exact words that set the budget signal.",
    "intuition_evidence": "Quote or paraphrase the words that drove the Hot/Lukewarm/Cold score.",
    "pain_point_evidence": "Quote or paraphrase the words that revealed each pain point."
  },

  "product_intent_detected": "Yes / No / Unclear",
  "products_identified": ["product name 1", "product name 2"],
  "estimated_deal_value": "High / Medium / Low / Unknown",
  "estimated_deal_value_range": "$X,000-$Y,000 or Unknown",

    "product_deal_evidence": {
    "product_intent_evidence": "Quote or paraphrase the exact words that show the customer wants a product.",
    "products_identified_evidence": "Quote or paraphrase the words that revealed each specific product.",
    "deal_value_evidence": "Quote or paraphrase the words that drove the High/Medium/Low/Unknown value tier.",
    "deal_value_range_evidence": "Quote or paraphrase the signals used to estimate the dollar range — company size, contract type, scope."
    }
}

Rules:
- Use ONLY the allowed labels shown above for each categorical field
- extracted_product_features.product_interest = what product capabilities they want
- extracted_product_features.support_interest = onboarding, implementation, account management, migration help, SLAs
  If no support interest is mentioned, return an empty list []
- intents must be short verb phrases e.g. ["evaluate vendors", "get leadership buy-in"]
- purchase_intent_reason must be ONE sentence naming the business trigger e.g. "CEO mandated vendor selection this month due to ERP integration deadlines"
- lead_scale combines company size AND deal scope: Enterprise = 200+ employees or org-wide rollout, Mid-Market = 50-200 employees or department-wide, SMB = under 50 employees, Startup = early stage or growth phase
- intuition_score: Hot = urgent + strong fit + decision maker present; Lukewarm = some signals but gaps exist; Cold = vague, no urgency, poor fit
- lead_quality_score: Strong = rich detail across all 4 fields; Moderate = some detail but gaps; Weak = sparse or generic text
- pain_points must describe the CURRENT problem, not the desired solution
- reasoning_evidence must reference actual words or phrases from the lead text, not invented descriptions
- type_of_customer: "Existing Customer" is NOT an option — leads are pre-filtered
- product_intent_detected: Yes = clear product want expressed; No = no product intent; Unclear = ambiguous signals
- products_identified = list the actual product names or types the lead is interested in — if none, return []
- estimated_deal_value: High = $50,000+; Medium = $10,000-$50,000; Low = under $10,000; Unknown = no signals
- estimated_deal_value_range = rough dollar range inferred from company size, deal scope, contract type, and budget signals — if no signals return "Unknown"
- product_deal_evidence must reference actual words or phrases from the lead text, not invented descriptions
- Do NOT add any keys not listed above
"""


# ======================================
# STEP 4: CALL OLLAMA
# ======================================

def extract_features_with_ollama(context: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": context}
        ],
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9}
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        raw_text = response.json()["message"]["content"].strip()

        if raw_text.startswith("```"):
            lines = [l for l in raw_text.split("\n") if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        return json.loads(raw_text)

    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_response": raw_text if 'raw_text' in locals() else ""}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Ollama. Run: ollama serve"}
    except Exception as e:
        return {"error": str(e)}


# ======================================
# STEP 5: VALIDATE
# ======================================

def validate_features(features: dict, lead_id: int) -> dict:
    warnings = []
    for field, allowed in ALLOWED_VALUES.items():
        val = features.get(field)
        if val and val not in allowed:
            warnings.append(f"{field}='{val}'")
    if warnings:
        print(f"    [WARN] Lead {lead_id} label issues: {'; '.join(warnings)}")
        features["_label_warnings"] = " | ".join(warnings)
    else:
        features["_label_warnings"] = ""
    return features


# ======================================
# STEP 6: FLATTEN
# Plain list fields → pipe-separated strings
# Nested dict fields → flattened with dot notation keys
# ======================================

LIST_FIELDS   = ["intents", "pain_points" , "products_identified"]
NESTED_FIELDS = ["extracted_product_features", "reasoning_evidence" , "product_deal_evidence"]

def flatten_features(features: dict) -> dict:
    flat = {}
    for key, value in features.items():

        if key in LIST_FIELDS and isinstance(value, list):
            flat[key] = " | ".join(str(v) for v in value)

        elif key in NESTED_FIELDS and isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, list):
                    flat[f"{key}__{sub_key}"] = " | ".join(str(v) for v in sub_val)
                else:
                    flat[f"{key}__{sub_key}"] = sub_val

        else:
            flat[key] = value

    return flat


# ======================================
# STEP 7: COLUMN DEFINITIONS
# ======================================

ORIGINAL_TEXT_COLS = [
    "customer_comments",
    "product_interest",
    "product_info",
    "lead_description",
]

NEW_FEATURE_COLS = [
    "intents",
    "lead_scale",
    "intuition_score",
    "extracted_product_features__product_interest",
    "extracted_product_features__support_interest",
    "pain_points",
    "purchase_intent_reason",
    "lead_quality_score",
    "product_intent_detected",
    "products_identified",
    "estimated_deal_value",
    "estimated_deal_value_range",
]

EXPLAINABILITY_COLS = [
    "reasoning",
    "reasoning_evidence__intent_level_evidence",
    "reasoning_evidence__urgency_evidence",
    "reasoning_evidence__budget_evidence",
    "reasoning_evidence__intuition_evidence",
    "reasoning_evidence__pain_point_evidence",
    "product_deal_evidence__product_intent_evidence",
    "product_deal_evidence__products_identified_evidence",
    "product_deal_evidence__deal_value_evidence",
    "product_deal_evidence__deal_value_range_evidence",
]


# ======================================
# STEP 8: PIPELINE
# ======================================

def run_pipeline():
    print("=" * 60)
    print("  Lead Feature Engineering Pipeline v3 — Ollama + Llama3")
    print("=" * 60)

    # Load from Excel — each column read separately, zipped per row
    # Data quality check runs automatically inside load_leads_from_excel
    df = load_leads_from_excel(LEADS_FILE)
    df["unified_lead_context"] = df.apply(build_unified_context, axis=1)
    print(f"Loaded {len(df)} leads from {LEADS_FILE} (MAX_LEADS={MAX_LEADS})\n")

    feature_rows = []

    for _, row in df.iterrows():
        lead_id = row["lead_id"]
        print(f"  Lead {lead_id:>3} ...", end=" ", flush=True)

        raw_features = extract_features_with_ollama(row["unified_lead_context"])

        time.sleep(0.5)

        if "error" in raw_features:
            print(f"ERROR — {raw_features['error']}")
            feature_rows.append({"lead_id": lead_id, "error": raw_features["error"]})
            continue

        raw_features = validate_features(raw_features, lead_id)
        flat = flatten_features(raw_features)
        flat["lead_id"] = lead_id

        print(
            f"OK  [{raw_features.get('qualification_status','?')}] "
            f"intuition={raw_features.get('intuition_score','?')} "
            f"scale={raw_features.get('lead_scale','?')} "
            f"quality={raw_features.get('lead_quality_score','?')}"
        )
        print(f"         intents: {flat.get('intents','?')}")
        print(f"         why buying: {raw_features.get('purchase_intent_reason','?')}")

        feature_rows.append(flat)

    features_df = pd.DataFrame(feature_rows)
    final_df    = df.merge(features_df, on="lead_id", how="left")

    # Drop helper column before export
    final_df = final_df.drop(columns=["empty_fields"], errors="ignore")

    # --- Full output (every column) ---
    final_df.to_excel("lead_features_output_v3.xlsx", index=False, engine="openpyxl")
    print(f"\nFull output saved      → lead_features_output_v3.xlsx ({len(final_df)} rows, {len(final_df.columns)} cols)")

    # --- Focused export: 4 original text + core feature columns ---
    focused_cols = ["lead_id"] + ORIGINAL_TEXT_COLS + NEW_FEATURE_COLS
    focused_cols = [c for c in focused_cols if c in final_df.columns]
    final_df[focused_cols].to_excel("leads_text_and_features_v3.xlsx", index=False, engine="openpyxl")
    print(f"Focused export saved   → leads_text_and_features_v3.xlsx ({len(focused_cols)} columns)")

    # --- Explainability export: why each decision was made ---
    explain_cols = ["lead_id"] + ORIGINAL_TEXT_COLS + EXPLAINABILITY_COLS
    explain_cols = [c for c in explain_cols if c in final_df.columns]
    final_df[explain_cols].to_excel("leads_explainability_v3.xlsx", index=False, engine="openpyxl")
    print(f"Explainability export  → leads_explainability_v3.xlsx ({len(explain_cols)} columns)")

    # --- Preview ---
    print("\n--- Feature preview (first 3 rows) ---")
    pd.set_option("display.max_colwidth", 45)
    preview = [c for c in ["lead_id"] + NEW_FEATURE_COLS if c in final_df.columns]
    print(final_df[preview].head(3).to_string(index=False))
    print("\nDone.")

    return final_df


if __name__ == "__main__":
    df = run_pipeline()