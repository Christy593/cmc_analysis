import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr
import statsmodels.api as sm


# 1. Load Survey


FILE = "AI_Ideation_2.xlsx" 

survey = pd.read_excel(FILE)

# clean column names
survey.columns = survey.columns.str.strip().str.lower()


# 2. Clean Data


survey = survey.rename(columns={"participant_id": "user_id"})
survey["user_id"] = survey["user_id"].astype(str)

survey = survey[
    survey["user_id"].notna() &
    (survey["user_id"].str.upper() != "N/A")
]

# convert numeric
for col in survey.columns:
    survey[col] = pd.to_numeric(survey[col], errors="coerce")


# 3. Helper


def reverse_7(x):
    if pd.isna(x):
        return np.nan
    return 8 - x

def paired_t(l1, l2):
    return ttest_rel(l1, l2).pvalue


# 4. Extract Survey Dimensions

def extract_construct(df, prefix, reverse_items=[]):
    mapping = {
        1: (f"{prefix}_2", f"{prefix}_3"),
        2: (f"{prefix}_5", f"{prefix}_6"),
        3: (f"{prefix}_8", f"{prefix}_9"),
    }

    l1_list = []
    l2_list = []

    for i, (l1_col, l2_col) in mapping.items():
        if l1_col not in df.columns:
            continue

        l1 = df[l1_col]
        l2 = df[l2_col]

        if i in reverse_items:
            l1 = l1.apply(reverse_7)
            l2 = l2.apply(reverse_7)

        l1_list.append(l1)
        l2_list.append(l2)

    l1_mean = pd.concat(l1_list, axis=1).mean(axis=1)
    l2_mean = pd.concat(l2_list, axis=1).mean(axis=1)

    return l1_mean, l2_mean

# 5. Build Survey Scores

survey_scores = pd.DataFrame()
survey_scores["user_id"] = survey["user_id"]

constructs = {
    "problem_understanding": {"prefix": "prob_understanding", "reverse": [2]},
    "idea_development": {"prefix": "idea_development", "reverse": [3]},
    "solution_evaluation": {"prefix": "solution_evaluation", "reverse": [3]},
    "efficiency": {"prefix": "efficiency", "reverse": [2,3]},
    "agency": {"prefix": "agency", "reverse": [1]}
}

for name, cfg in constructs.items():
    l1, l2 = extract_construct(survey, cfg["prefix"], cfg["reverse"])

    survey_scores[f"{name}_L1"] = l1
    survey_scores[f"{name}_L2"] = l2
    survey_scores[f"{name}_diff"] = l2 - l1


# 6. Proficiency（重点）

if "years using english" in survey.columns:
    survey_scores["english_years"] = survey["years using english"]


# 7. AI Usage

if "how frequently do you use ai for help with writing assignments?" in survey.columns:
    survey_scores["ai_usage"] = survey[
        "how frequently do you use ai for help with writing assignments?"
    ]


# 8. Preference


if "overall_preference" in survey.columns:
    survey_scores["preference"] = survey["overall_preference"]

print("\n=== L1 vs L2 Comparison ===")

for m in constructs.keys():
    l1 = survey_scores[f"{m}_L1"]
    l2 = survey_scores[f"{m}_L2"]

    mask = (~l1.isna()) & (~l2.isna())

    if mask.sum() > 2:
        p = paired_t(l1[mask], l2[mask])
        diff = (l2 - l1).mean()

        print(f"{m}: diff(L2-L1)={diff:.2f}, p={p:.3f}")


print("\n=== Regression: proficiency ===")

if "english_years" in survey_scores.columns:

    for m in constructs.keys():
        df = survey_scores[[f"{m}_diff", "english_years"]].dropna()

        if len(df) > 3:
            X = sm.add_constant(df["english_years"])
            y = df[f"{m}_diff"]

            model = sm.OLS(y, X).fit()

            print(f"\n--- {m} ---")
            print(model.summary())



print("\n=== Language Role Pattern ===")

for m in constructs.keys():
    diff = survey_scores[f"{m}_diff"].mean()
    print(f"{m}: L2 advantage = {diff:.2f}")




print("\n=== Correlation between dimensions ===")

if "efficiency_L2" in survey_scores.columns:

    for m in ["idea_development_L2", "problem_understanding_L2"]:
        if m in survey_scores.columns:

            x = survey_scores["efficiency_L2"]
            y = survey_scores[m]

            mask = (~x.isna()) & (~y.isna())

            if mask.sum() > 3:
                r, p = pearsonr(x[mask], y[mask])
                print(f"efficiency vs {m}: r={r:.2f}, p={p:.3f}")

import matplotlib.pyplot as plt
import numpy as np

metrics = [
    "Problem\nUnderstanding",
    "Idea\nDevelopment",
    "Solution\nEvaluation",
    "Efficiency",
    "Agency"
]

raw_metrics = [
    "problem_understanding",
    "idea_development",
    "solution_evaluation",
    "efficiency",
    "agency"
]

values = [
    survey_scores[f"{m}_diff"].mean()
    for m in raw_metrics
]


colors = [
    "#A7D3F2" if v < 0 else "#1F77B4"
    for v in values
]

plt.figure(figsize=(8,5))

bars = plt.bar(metrics, values, color=colors)

# baseline
plt.axhline(0, linestyle="--", linewidth=1, color="#888888")

# 数值标注
for i, v in enumerate(values):
    plt.text(i, v + (0.03 if v >= 0 else -0.06),
             f"{v:.2f}",
             ha="center",
             fontsize=10)

for i, v in enumerate(values):
    label = "L2" if v > 0 else "L1"
    plt.text(i, v/2,
             label,
             ha="center",
             fontsize=11,
             color="black",
             fontweight="bold")

plt.ylabel("Difference (L2 − L1)")
plt.title("Language Roles in AI Ideation")

plt.tight_layout()
plt.savefig("language_role_blue.png", dpi=300)
plt.show()


from scipy.stats import pearsonr
import numpy as np

x = survey_scores["problem_understanding_L2"]
y = survey_scores["efficiency_L2"]

mask = (~x.isna()) & (~y.isna())

r, p = pearsonr(x[mask], y[mask])

plt.figure(figsize=(5,5))


plt.scatter(x[mask], y[mask], color="#A7D3F2")


z = np.polyfit(x[mask], y[mask], 1)
p_line = np.poly1d(z)

plt.plot(x[mask], p_line(x[mask]), color="#1F77B4", linewidth=2)

plt.xlabel("Problem Understanding (L2)")
plt.ylabel("Efficiency (L2)")

plt.title(f"Understanding → Efficiency\nr = {r:.2f}, p = {p:.03f}")

plt.tight_layout()
plt.savefig("correlation_blue.png", dpi=300)
plt.show()


plt.figure(figsize=(5,4))

diff = survey_scores["problem_understanding_diff"]


plt.hist(diff, bins=6, color="#A7D3F2", edgecolor="#1F77B4")

# baseline
plt.axvline(0, linestyle="--", color="#888888")


mean_val = diff.mean()
plt.axvline(mean_val, linestyle=":", color="#1F77B4", linewidth=2)

plt.xlabel("Difference (L2 − L1)")
plt.ylabel("Count")

plt.title("Problem Understanding Difference")

plt.tight_layout()
plt.savefig("diff_blue.png", dpi=300)
plt.show()