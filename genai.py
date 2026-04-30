import pandas as pd
import random
import json
import requests

# ======================================
# CONFIG
# ======================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
NUM_SYNTHETIC_LEADS = 25


# ======================================
# STEP 1: SYNTHETIC DATASET (LONGER TEXT)
# ======================================

customer_comments_samples = [
    "We are currently reviewing multiple CRM vendors because our existing system is outdated and reporting is extremely slow. Our sales leadership wants better analytics visibility before the next quarter planning cycle. Pricing and implementation timelines are important because we need a decision within the next two weeks.",

    "Our CEO asked us to urgently evaluate enterprise solutions that can integrate with our current ERP and finance systems. We need API access, workflow automation, and advanced reporting. This is a strategic purchase and we are trying to finalize vendors this month.",

    "We are just starting to research options for customer support and analytics platforms. There is no immediate timeline yet because internal approvals are still pending. We mainly want to understand pricing models and what vendors are available in the market.",

    "Budget is a concern for us because we are a smaller team and cannot afford enterprise pricing. However, we still need reporting dashboards and customer workflow automation. We are comparing affordable vendors before making any commitment.",

    "Our procurement team is evaluating 3 vendors for a company-wide rollout across operations, finance, and sales departments. We need strong implementation support and long-term partnership potential. Final approval will come from the department head and CFO."
]

product_interest_samples = [
    "CRM platform with advanced analytics, reporting dashboards, and workflow automation for enterprise sales teams.",
    "Enterprise API integration package with ERP connectivity, finance reporting, and compliance workflows.",
    "Customer support platform with ticketing, reporting, and automation for service operations.",
    "Sales dashboard and reporting suite for leadership visibility and performance monitoring.",
    "Mid-market automation platform for reporting, operations efficiency, and internal process management."
]

product_info_samples = [
    "Enterprise annual plan with advanced integrations, API access, implementation consulting, and dedicated account management.",
    "Mid-market package with reporting features, workflow automation, and cloud deployment for regional teams.",
    "Starter package for smaller businesses focused on dashboard visibility and process improvements with limited integrations.",
    "Custom implementation required due to legacy systems and compliance requirements across multiple departments.",
    "Annual contract preferred with onboarding support, migration assistance, and leadership reporting features included."
]

lead_description_samples = [
    "Mid-size finance company with 200 employees currently replacing legacy systems and evaluating long-term technology partners.",
    "Large healthcare enterprise operating across multiple countries with centralized procurement and strong compliance requirements.",
    "Startup with 10 employees exploring affordable software options while validating operational requirements and future growth plans.",
    "Retail business expanding into multiple locations and looking for stronger reporting visibility across customer operations.",
    "Manufacturing company modernizing internal systems with procurement team comparing vendors for enterprise rollout."
]


def generate_synthetic_leads(n=NUM_SYNTHETIC_LEADS):
    rows = []

    for i in range(1, n + 1):
        rows.append({
            "lead_id": i,
            "customer_comments": random.choice(customer_comments_samples),
            "product_interest": random.choice(product_interest_samples),
            "product_info": random.choice(product_info_samples),
            "lead_description": random.choice(lead_description_samples)
        })

    return pd.DataFrame(rows)


# ======================================
# STEP 2: COMBINE TEXT COLUMNS
# ======================================


def build_unified_context(row):
    return f"""
Customer Comments:
{row['customer_comments']}

Product Interest:
{row['product_interest']}

Product Info:
{row['product_info']}

Lead Description:
{row['lead_description']}
""".strip()


df = generate_synthetic_leads()
df["unified_lead_context"] = df.apply(build_unified_context, axis=1)
df.to_csv("synthetic_leads.csv", index=False)

print("Synthetic dataset created: synthetic_leads.csv")


# ======================================
# STEP 3: LLM SIGNAL EXTRACTION (JSON)
# ======================================


def analyze_lead_with_llama(context):
    prompt = f"""
You are a B2B lead qualification assistant.

Analyze the lead and return STRICT JSON ONLY using this exact structure:

{{
  "intent_level": "Low / Medium / High",
  "timeline": "Given / Not Given / Urgent",
  "budget": "Given / Not Given",
  "type_of_customer": "Willing to Buy / Researching / Existing Customer / Student / Other",
  "customer_value": "High / Medium / Low",
  "product_match": "Strong / Partial / Weak",
  "pain_point_clarity": "Clear / Moderate / Missing",
  "missing_information": ["list missing areas here"]
}}

Rules:
- Do not explain outside JSON
- If uncertain, choose the safest option
- Detect if follow-up is needed based on missing details

Lead Context:
{context}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return json.loads(response.json()["response"])

    except Exception as e:
        return {"error": str(e)}


# ======================================
# STEP 4: RULE-BASED SCORING
# ======================================


def score_lead(result):
    score = 0

    intent_scores = {
        "High": 25,
        "Medium": 15,
        "Low": 5
    }

    timeline_scores = {
        "Urgent": 20,
        "Given": 10,
        "Not Given": 0
    }

    customer_value_scores = {
        "High": 20,
        "Medium": 10,
        "Low": 5
    }

    product_match_scores = {
        "Strong": 20,
        "Partial": 10,
        "Weak": 0
    }

    pain_point_scores = {
        "Clear": 15,
        "Moderate": 8,
        "Missing": 0
    }

    score += intent_scores.get(result.get("intent_level"), 0)
    score += timeline_scores.get(result.get("timeline"), 0)
    score += customer_value_scores.get(result.get("customer_value"), 0)
    score += product_match_scores.get(result.get("product_match"), 0)
    score += pain_point_scores.get(result.get("pain_point_clarity"), 0)

    if result.get("budget") == "Given":
        score += 10

    if result.get("type_of_customer") == "Willing to Buy":
        score += 15
    elif result.get("type_of_customer") == "Researching":
        score += 5

    if score >= 75:
        status = "Qualified"
    elif score >= 50:
        status = "Needs Follow-up"
    else:
        status = "Low Priority"

    return score, status


# ======================================
# STEP 5: RUN PIPELINE
# ======================================

results = []

for _, row in df.head(5).iterrows():
    print(f"Analyzing Lead ID: {row['lead_id']}")

    llm_result = analyze_lead_with_llama(row["unified_lead_context"])

    if "error" in llm_result:
        final_score = 0
        status = "Error"
    else:
        final_score, status = score_lead(llm_result)

    results.append({
        "lead_id": row["lead_id"],
        "llm_result": json.dumps(llm_result),
        "final_score": final_score,
        "qualification_status": status,
        "follow_up_needed": llm_result.get("missing_information", []) if isinstance(llm_result, dict) else []
    })


results_df = pd.DataFrame(results)
results_df.to_csv("lead_qualification_results.csv", index=False)

print("Results saved to: lead_qualification_results.csv")
print(results_df)
