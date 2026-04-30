import pandas as pd
import random
import json
import requests

# ======================================
# CONFIG
# ======================================

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"
NUM_SYNTHETIC_LEADS = 20


# ======================================
# STEP 1: SYNTHETIC DATASET (TEXT-HEAVY)
# ======================================

customer_comments_samples = [
    "We are reviewing CRM vendors because our current system is outdated and reporting is slow. Leadership wants better visibility before the next planning cycle. Pricing and implementation speed are important because we may need a decision within two weeks.",

    "Our CEO requested an urgent evaluation of enterprise solutions that can integrate with our ERP and finance systems. We need API access, workflow automation, and compliance reporting. Final vendor selection should happen this month.",

    "We are still researching available options for analytics and customer support platforms. There is no immediate timeline yet because internal approvals are pending. We mainly want to understand pricing and available vendors.",

    "Budget is a major concern since we are a smaller company. We still need dashboards and workflow automation, but enterprise pricing may be difficult for us. We are comparing affordable vendors before moving forward.",

    "Our procurement team is comparing three vendors for a company-wide rollout across finance, operations, and sales. We need implementation support and long-term partnership potential. Final approval will come from leadership and finance."
]

product_interest_samples = [
    "CRM platform with analytics dashboards, workflow automation, and enterprise reporting.",
    "Enterprise API integration package with ERP connectivity and compliance reporting.",
    "Customer support platform with ticketing, reporting, and automation.",
    "Sales dashboard and reporting suite for leadership visibility.",
    "Mid-market operations automation platform with reporting tools."
]

product_info_samples = [
    "Enterprise annual plan with advanced integrations, onboarding support, and dedicated account management.",
    "Mid-market package with workflow automation, reporting features, and cloud deployment.",
    "Starter package for smaller businesses focused on dashboards and process improvements.",
    "Custom implementation required due to legacy systems and compliance requirements.",
    "Annual contract preferred with migration assistance and leadership reporting features included."
]

lead_description_samples = [
    "Mid-size finance company with 200 employees replacing legacy systems and evaluating long-term technology partners.",
    "Large healthcare enterprise across multiple countries with centralized procurement and compliance requirements.",
    "Startup with 10 employees exploring affordable software options while validating future growth plans.",
    "Retail business expanding into multiple locations and looking for stronger reporting visibility.",
    "Manufacturing company modernizing internal systems with procurement comparing vendors for enterprise rollout."
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
# STEP 2: BUILD UNIFIED LEAD CONTEXT
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


# ======================================
# STEP 3: CHAT COMPLETION WITH LLAMA 3
# ======================================


def analyze_lead_with_llama(context):
    system_prompt = """
You are a B2B lead qualification assistant.

Return STRICT JSON ONLY using this exact structure:

{
  "intent_level": "Low / Medium / High",
  "timeline": "Urgent / Given / Not Given",
  "budget": "Given / Not Given",
  "type_of_customer": "Willing to Buy / Researching / Existing Customer / Student / Other",
  "customer_value": "High / Medium / Low",
  "product_match": "Strong / Partial / Weak",
  "pain_point_clarity": "Clear / Moderate / Missing",
  "missing_information": ["list of missing details"],
  "sufficient_information": "Yes / No",
  "confidence_level": "High / Medium / Low",
  "reasoning": "short explanation",
  "qualification_status": "Qualified / Needs Follow-Up / Not Qualified"
}

Rules:
- Use only allowed labels
- No explanation outside JSON
- If important details are missing, sufficient_information must be No
- If enough info exists but poor fit, qualification_status can be Not Qualified
- missing_information must clearly explain what follow-up is needed
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": context
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        raw_response = response.json()["message"]["content"]

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON returned by model",
                "raw_response": raw_response
            }

    except Exception as e:
        return {
            "error": str(e)
        }


# ======================================
# STEP 4: RUN PIPELINE
# ======================================


def main():
    df = generate_synthetic_leads()
    df["unified_lead_context"] = df.apply(build_unified_context, axis=1)

    df.to_csv("synthetic_leads.csv", index=False)
    print("Saved: synthetic_leads.csv")

    results = []

    for _, row in df.head(5).iterrows():
        print(f"Analyzing Lead ID: {row['lead_id']}")

        llm_result = analyze_lead_with_llama(row["unified_lead_context"])

        results.append({
            "lead_id": row["lead_id"],
            "customer_comments": row["customer_comments"],
            "qualification_json": json.dumps(llm_result, ensure_ascii=False)
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("lead_qualification_results_v2.csv", index=False)

    print("Saved: lead_qualification_results_v2.csv")
    print(results_df[["lead_id", "qualification_json"]])


if __name__ == "__main__":
    main()
