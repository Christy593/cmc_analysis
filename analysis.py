import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, pearsonr
import matplotlib.pyplot as plt
import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import itertools



# 1. Load Data

survey = pd.read_csv("data/clean_survey.csv")
logs = pd.read_csv("data/logs.csv")


# 2. Clean Survey Data

# Example expected survey columns:
# participant_id
# problem_understanding_L1, problem_understanding_EN
# idea_development_L1, idea_development_EN
# solution_evaluation_L1, solution_evaluation_EN
# efficiency_L1, efficiency_EN
# cognitive_load_L1, cognitive_load_EN
# agency_L1, agency_EN
# satisfaction_L1, satisfaction_EN
# preferred_condition

survey = survey.dropna(subset=["participant_id"])

metrics = [
    "problem_understanding",
    "idea_development",
    "solution_evaluation",
    "efficiency",
    "cognitive_load",
    "agency",
    "satisfaction"
]


# 3. Paired Comparison: L1 vs English


results = []

for metric in metrics:
    l1_col = f"{metric}_L1"
    en_col = f"{metric}_EN"

    df = survey[[l1_col, en_col]].dropna()

    l1_mean = df[l1_col].mean()
    en_mean = df[en_col].mean()

    diff = df[l1_col] - df[en_col]

    # Paired t-test
    t_stat, p_value = ttest_rel(df[l1_col], df[en_col])

    # Wilcoxon test, safer for small sample size
    try:
        w_stat, w_p = wilcoxon(df[l1_col], df[en_col])
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    results.append({
        "Metric": metric,
        "L1 Mean": round(l1_mean, 3),
        "English Mean": round(en_mean, 3),
        "Mean Difference": round(diff.mean(), 3),
        "Paired t-test p": round(p_value, 4),
        "Wilcoxon p": round(w_p, 4)
    })

results_df = pd.DataFrame(results)
print("\n=== Paired Comparison Results ===")
print(results_df)

results_df.to_csv("paired_comparison_results.csv", index=False)


# 4. Plot L1 vs English Mean Ratings

plot_data = []

for metric in metrics:
    plot_data.append({
        "Metric": metric,
        "Condition": "Native Language",
        "Mean": survey[f"{metric}_L1"].mean()
    })
    plot_data.append({
        "Metric": metric,
        "Condition": "English",
        "Mean": survey[f"{metric}_EN"].mean()
    })

plot_df = pd.DataFrame(plot_data)

plt.figure(figsize=(12, 6))

for condition in ["Native Language", "English"]:
    subset = plot_df[plot_df["Condition"] == condition]
    plt.plot(subset["Metric"], subset["Mean"], marker="o", label=condition)

plt.title("Mean Survey Ratings: Native Language vs English")
plt.xlabel("Survey Dimension")
plt.ylabel("Mean Rating")
plt.xticks(rotation=30, ha="right")
plt.ylim(1, 7)
plt.legend()
plt.tight_layout()
plt.savefig("survey_l1_vs_english.png", dpi=300)
plt.show()



# 5. Clean Log Data

# Example expected log columns:
# participant_id
# condition: L1 or EN
# prompt_count
# regeneration_count
# start_time
# end_time
# outline_text

logs["outline_text"] = logs["outline_text"].fillna("")

logs["outline_word_count"] = logs["outline_text"].apply(lambda x: len(str(x).split()))
logs["outline_char_count"] = logs["outline_text"].apply(lambda x: len(str(x)))

# Optional readability score
logs["readability_score"] = logs["outline_text"].apply(
    lambda x: textstat.flesch_reading_ease(str(x)) if len(str(x).split()) > 10 else np.nan
)



# 6. Behavioral Comparison


behavior_metrics = [
    "prompt_count",
    "regeneration_count",
    "outline_word_count",
    "outline_char_count",
    "readability_score"
]

behavior_results = []

for metric in behavior_metrics:
    pivot = logs.pivot(index="participant_id", columns="condition", values=metric).dropna()

    if "L1" in pivot.columns and "EN" in pivot.columns:
        t_stat, p_value = ttest_rel(pivot["L1"], pivot["EN"])

        behavior_results.append({
            "Behavior Metric": metric,
            "L1 Mean": round(pivot["L1"].mean(), 3),
            "English Mean": round(pivot["EN"].mean(), 3),
            "Mean Difference": round((pivot["L1"] - pivot["EN"]).mean(), 3),
            "p-value": round(p_value, 4)
        })

behavior_df = pd.DataFrame(behavior_results)

print("\n=== Behavioral Comparison Results ===")
print(behavior_df)

behavior_df.to_csv("behavioral_comparison_results.csv", index=False)



# 7. Plot Behavioral Metrics


for metric in behavior_metrics:
    summary = logs.groupby("condition")[metric].mean().reset_index()

    plt.figure(figsize=(6, 4))
    plt.bar(summary["condition"], summary[metric])
    plt.title(f"Average {metric}: L1 vs English")
    plt.xlabel("Condition")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png", dpi=300)
    plt.show()


# =========================
# 8. Preference Count
# =========================

if "preferred_condition" in survey.columns:
    preference_counts = survey["preferred_condition"].value_counts()

    print("\n=== Preference Counts ===")
    print(preference_counts)

    plt.figure(figsize=(6, 4))
    preference_counts.plot(kind="bar")
    plt.title("Overall Language Preference")
    plt.xlabel("Preferred Condition")
    plt.ylabel("Number of Participants")
    plt.tight_layout()
    plt.savefig("language_preference.png", dpi=300)
    plt.show()


# =========================
# 9. Correlation Analysis
# =========================

# Example: prompt count vs satisfaction
merged = logs.merge(survey, on="participant_id", how="left")

correlation_results = []

for condition in ["L1", "EN"]:
    subset = merged[merged["condition"] == condition]

    satisfaction_col = f"satisfaction_{condition}"

    if satisfaction_col in subset.columns:
        temp = subset[["prompt_count", "regeneration_count", "outline_word_count", satisfaction_col]].dropna()

        for behavior in ["prompt_count", "regeneration_count", "outline_word_count"]:
            if len(temp) > 2:
                r, p = pearsonr(temp[behavior], temp[satisfaction_col])

                correlation_results.append({
                    "Condition": condition,
                    "Behavior": behavior,
                    "Survey Outcome": satisfaction_col,
                    "Correlation r": round(r, 3),
                    "p-value": round(p, 4)
                })

correlation_df = pd.DataFrame(correlation_results)

print("\n=== Correlation Results ===")
print(correlation_df)

correlation_df.to_csv("correlation_results.csv", index=False)


# =========================
# 10. Simple Qualitative Coding Helper
# =========================

if "open_ended_response" in survey.columns:
    open_responses = survey[["participant_id", "open_ended_response"]].dropna()
    open_responses.to_csv("open_ended_responses_for_coding.csv", index=False)

    print("\nOpen-ended responses exported for qualitative coding.")


# 11. Cosine Similarity Analysis 

print("\nRunning Cosine Similarity Analysis ")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

similarity_results = []

# divide by prompt 
for prompt in logs["prompt"].unique():

    subset_prompt = logs[logs["prompt"] == prompt]

    for condition in ["L1", "EN"]:
        subset = subset_prompt[subset_prompt["condition"] == condition]

        texts = subset["outline_text"].dropna().tolist()

        # 至少要有2个才有意义
        if len(texts) < 2:
            continue

        # 生成 embeddings
        embeddings = model.encode(texts)

        # 计算 pairwise similarity
        sim_matrix = cosine_similarity(embeddings)

        # 取上三角（避免重复和自己跟自己比）
        sim_scores = []

        for i, j in itertools.combinations(range(len(texts)), 2):
            sim_scores.append(sim_matrix[i][j])

        avg_similarity = np.mean(sim_scores)

        similarity_results.append({
            "Prompt": prompt,
            "Condition": condition,
            "Avg Cosine Similarity": round(avg_similarity, 4),
            "Num Samples": len(texts)
        })

similarity_df = pd.DataFrame(similarity_results)

print("\n=== Cosine Similarity Results ===")
print(similarity_df)

similarity_df.to_csv("cosine_similarity_results.csv", index=False)