BASIC_TEMPLATE = """Instruction:
Based on the user's preferences in the domain {source_domain}, generate personalized recommendations for items from the domain {target_domain}.

Input:
User's highly-rated items from domain {source_domain}:
{user_history}

Target domain: {target_domain}

Task:
Recommend the top {num_recs} {target_domain} items the user is most likely to enjoy, based on their preferences above. Provide a brief explanation for each recommendation that connects it to the user's interests.

Output Format:
1. Item Title – Explanation
2. ...
3. ...
"""

ENHANCED_TEMPLATE = """You are an expert cross-domain recommendation engine. Your goal is to generate high-quality recommendations.

**Source Domain:** {source_domain}
**User Preferences:**
{user_history}

**Target Domain:** {target_domain}
**Number of Recommendations:** {num_recs}

**Recommendation Criteria:**
- Identify patterns in the user's preferences.
- Suggest items in the target domain that align with those patterns.
- Provide a brief rationale for each recommendation.

**Output Format:**
1. **[Item Title]** – Explanation
2. **[Item Title]** – Explanation
3. **[Item Title]** – Explanation
"""

EVALUATION_TEMPLATE = """Instruction:
Given the user's liked items in {source_domain}, assess whether they would enjoy the following item from {target_domain}:

**Candidate Item:** {target_item}

**User's Preferences in {source_domain}:**
{user_history}

**Task:**
- Analyze alignment between the candidate item and user preferences.
- Return a judgment (Yes/No) with a short explanation.

Answer Format:
Prediction: Yes/No
Confidence: High/Medium/Low
Reasoning: Explanation
"""

def get_template(template_type="basic"):
    if template_type == "enhanced":
        return ENHANCED_TEMPLATE
    elif template_type == "evaluation":
        return EVALUATION_TEMPLATE
    else:
        return BASIC_TEMPLATE
