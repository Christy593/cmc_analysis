import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

LOG_FILE = "output.csv"
PART_FILE = "output_participants.csv"
SURVEY_FILE = "AI_Ideation.xlsx"


# 1. Read files

logs = pd.read_csv(LOG_FILE)
parts = pd.read_csv(PART_FILE)
survey = pd.read_excel(SURVEY_FILE)

for df in [logs, parts, survey]:
    df.columns = df.columns.str.strip().str.lower()

logs["user_id"] = logs["user_id"].astype(str)
parts["user_id"] = parts["user_id"].astype(str)


# 2. Clean participant output data

parts = parts[
    parts["user_id"].notna()
    & (parts["user_id"].str.upper() != "N/A")
    & (parts["completed"] == 1)
].copy()


# 3. Helper functions

def count_words(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r"\b\w+\b", str(text), flags=re.UNICODE))

def count_chars(text):
    if pd.isna(text):
        return 0
    return len(str(text))

def count_solutions(text):
    if pd.isna(text):
        return 0
    text = str(text).lower()
    patterns = [
        r"\bsolution\s*1\b", r"\bsolution\s*2\b",
        r"\b1[\.\)]", r"\b2[\.\)]",
        r"\bfirst\b", r"\bsecond\b",
        r"첫 번째", r"두 번째",
        r"thứ nhất", r"thứ hai"
    ]
    count = sum(len(re.findall(p, text)) for p in patterns)
    return min(count, 4)

def has_tradeoff(text):
    if pd.isna(text):
        return 0

    text = str(text).lower()

    # 强信号
    strong_keywords = [
        # English
        "trade-off", "tradeoff", "limitation", "drawback", "downside", "risk", "cost",

        # Chinese
        "缺点", "局限", "代价", "风险", "问题",

        # Korean
        "문제점", "한계", "단점",

        # Vietnamese
        "hạn chế", "nhược điểm", "rủi ro"
    ]


    contrast_words = [
        "however", "but", "on the other hand",
        "但是", "然而",
        "하지만",
        "tuy nhiên"
    ]


    if any(k in text for k in strong_keywords):
        return 1

    if any(w in text for w in contrast_words):

        if ("more" in text and "less" in text) or ("increase" in text and "decrease" in text):
            return 1

    return 0

def reverse_7(x):
    if pd.isna(x):
        return np.nan
    return 8 - x

def paired_test(l1, l2):
    diff = l1 - l2
    result = {
        "L1_mean": l1.mean(),
        "L2_mean": l2.mean(),
        "mean_diff_L1_minus_L2": diff.mean(),
        "n": len(l1)
    }

    if len(l1) >= 2 and diff.std(ddof=1) != 0:
        result["paired_t_p"] = ttest_rel(l1, l2).pvalue
    else:
        result["paired_t_p"] = np.nan

    try:
        if (diff != 0).any():
            result["wilcoxon_p"] = wilcoxon(l1, l2).pvalue
        else:
            result["wilcoxon_p"] = np.nan
    except Exception:
        result["wilcoxon_p"] = np.nan

    return result

# Behavior analysis from logs

logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
logs = logs.sort_values(["user_id", "timestamp", "id"]).copy()

# A new task starts when user message starts with "Task prompt:"
logs["is_task_start"] = (
    (logs["role"].str.lower() == "user")
    & logs["content"].astype(str).str.startswith("Task prompt:")
)

logs["task_order"] = logs.groupby("user_id")["is_task_start"].cumsum()
logs = logs[logs["task_order"] > 0].copy()

user_msgs = logs[logs["role"].str.lower() == "user"].copy()
assistant_msgs = logs[logs["role"].str.lower() == "assistant"].copy()

behavior = logs.groupby(["user_id", "task_order"]).agg(
    start_time=("timestamp", "min"),
    end_time=("timestamp", "max"),
    total_messages=("id", "count")
).reset_index()

prompt_metrics = user_msgs.groupby(["user_id", "task_order"]).agg(
    prompt_count=("id", "count"),
    user_words=("content", lambda x: sum(count_words(v) for v in x)),
    user_chars=("content", lambda x: sum(count_chars(v) for v in x)),
    avg_prompt_words=("content", lambda x: np.mean([count_words(v) for v in x])),
).reset_index()

assistant_metrics = assistant_msgs.groupby(["user_id", "task_order"]).agg(
    assistant_count=("id", "count"),
    assistant_words=("content", lambda x: sum(count_words(v) for v in x)),
).reset_index()

behavior = behavior.merge(prompt_metrics, on=["user_id", "task_order"], how="left")
behavior = behavior.merge(assistant_metrics, on=["user_id", "task_order"], how="left")

behavior["duration_seconds"] = (
    behavior["end_time"] - behavior["start_time"]
).dt.total_seconds()

if "edit_index" in logs.columns:
    regen = logs[
        logs["edit_index"].notna()
    ].groupby(["user_id", "task_order"]).size().reset_index(name="regenerate_count")
    behavior = behavior.merge(regen, on=["user_id", "task_order"], how="left")
else:
    behavior["regenerate_count"] = 0

behavior = behavior.fillna(0)


# Final draft output analysis

parts["draft_words"] = parts["final_draft"].apply(count_words)
parts["draft_chars"] = parts["final_draft"].apply(count_chars)
parts["solution_count"] = parts["final_draft"].apply(count_solutions)
parts["has_tradeoff"] = parts["final_draft"].apply(has_tradeoff)

task_df = parts.merge(
    behavior,
    on=["user_id", "task_order"],
    how="left"
).fillna(0)

task_df["condition"] = np.where(
    task_df["language"].str.lower() == "english",
    "L2_English",
    "L1_Native"
)

#task_df.to_excel("task_level_behavior_output.xswl", index=False)


# Paired L1 vs L2 behavior/output

metrics = [
    "prompt_count",
    "avg_prompt_words",
    "user_words",
    "assistant_count",
    "assistant_words",
    "duration_seconds",
    "regenerate_count",
    "draft_words",
    "draft_chars",
    "solution_count",
    "has_tradeoff"
]

paired_rows = []

for uid, group in task_df.groupby("user_id"):
    l1 = group[group["condition"] == "L1_Native"]
    l2 = group[group["condition"] == "L2_English"]

    if len(l1) == 1 and len(l2) == 1:
        row = {"user_id": uid}
        for m in metrics:
            row[f"{m}_L1"] = l1[m].values[0]
            row[f"{m}_L2"] = l2[m].values[0]
            row[f"{m}_diff"] = l1[m].values[0] - l2[m].values[0]
        paired_rows.append(row)

paired_behavior = pd.DataFrame(paired_rows)
# paired_behavior.to_csv("paired_behavior_output.csv", index=False)

behavior_results = []

for m in metrics:
    if f"{m}_L1" in paired_behavior.columns:
        res = paired_test(
            paired_behavior[f"{m}_L1"],
            paired_behavior[f"{m}_L2"]
        )
        res["metric"] = m
        behavior_results.append(res)

behavior_results = pd.DataFrame(behavior_results)
# behavior_results.to_csv("behavior_output_stats.csv", index=False)

# Survey analysis

survey = survey.rename(columns={"participant_id": "user_id"})
survey["user_id"] = survey["user_id"].astype(str)

# remove Qualtrics question-text row
survey = survey[
    survey["user_id"].notna()
    & ~survey["user_id"].str.contains("What is your SONA ID", na=False)
    & (survey["user_id"].str.upper() != "N/A")
].copy()

# convert survey rating columns to numeric
for col in survey.columns:
    if re.search(r"_(\d+)$", col):
        survey[col] = pd.to_numeric(survey[col], errors="coerce")

# In Qualtrics export pattern:
# each construct has 9 columns:
# item text col, Native rating, English rating, repeated 3 times.
constructs = {
    "problem_understanding": {
        "prefix": "prob_understanding",
        "reverse_items": [2]
    },
    "idea_development": {
        "prefix": "idea_development",
        "reverse_items": [3]
    },
    "solution_evaluation": {
        "prefix": "solution_evaluation",
        "reverse_items": [3]
    },
    "efficiency": {
        "prefix": "efficiency",
        "reverse_items": [2, 3]
    },
    "agency": {
        "prefix": "agency",
        "reverse_items": [1]
    }
}

survey_scores = survey[["user_id", "overall_preference", "explanation"]].copy()

for construct, info in constructs.items():
    prefix = info["prefix"]
    reverse_items = info["reverse_items"]

    l1_items = []
    l2_items = []

    # item 1: _2 native, _3 english
    # item 2: _5 native, _6 english
    # item 3: _8 native, _9 english
    mapping = {
        1: (f"{prefix}_2", f"{prefix}_3"),
        2: (f"{prefix}_5", f"{prefix}_6"),
        3: (f"{prefix}_8", f"{prefix}_9"),
    }

    for item_num, (l1_col, l2_col) in mapping.items():
        if l1_col not in survey.columns or l2_col not in survey.columns:
            continue

        l1_series = survey[l1_col].copy()
        l2_series = survey[l2_col].copy()

        if item_num in reverse_items:
            l1_series = l1_series.apply(reverse_7)
            l2_series = l2_series.apply(reverse_7)

        survey_scores[f"{construct}_item{item_num}_L1"] = l1_series
        survey_scores[f"{construct}_item{item_num}_L2"] = l2_series

        l1_items.append(f"{construct}_item{item_num}_L1")
        l2_items.append(f"{construct}_item{item_num}_L2")

    survey_scores[f"{construct}_L1"] = survey_scores[l1_items].mean(axis=1)
    survey_scores[f"{construct}_L2"] = survey_scores[l2_items].mean(axis=1)
    survey_scores[f"{construct}_diff"] = (
        survey_scores[f"{construct}_L1"] - survey_scores[f"{construct}_L2"]
    )

# survey_scores.to_csv("survey_scores.csv", index=False)

survey_results = []

for construct in constructs.keys():
    l1_col = f"{construct}_L1"
    l2_col = f"{construct}_L2"

    temp = survey_scores[[l1_col, l2_col]].dropna()

    if len(temp) > 0:
        res = paired_test(temp[l1_col], temp[l2_col])
        res["metric"] = construct
        survey_results.append(res)

survey_results = pd.DataFrame(survey_results)
#  survey_results.to_csv("survey_stats.csv", index=False)


# Integrated user-level file

integrated = paired_behavior.merge(
    survey_scores,
    on="user_id",
    how="left"
)

#  integrated.to_csv("integrated_final_analysis.csv", index=False)


# Plots 


def paired_scatter(df, metric, title, filename):
    l1 = df[f"{metric}_L1"]
    l2 = df[f"{metric}_L2"]

    plt.figure(figsize=(5,5))
    plt.scatter(l1, l2)


    max_val = max(max(l1), max(l2))
    plt.plot([0, max_val], [0, max_val], linestyle='--')

    plt.xlabel("L1 Native")
    plt.ylabel("L2 English")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# draft 

# Prompt Count
paired_scatter(
    paired_behavior,
    "prompt_count",
    "Prompt Count (Paired)",
    "scatter_prompt_count.png"
)

# Draft Words
paired_scatter(
    paired_behavior,
    "draft_words",
    "Draft Words (Paired)",
    "scatter_draft_words.png"
)

# Efficiency
paired_scatter(
    survey_scores,
    "efficiency",
    "Efficiency (Paired)",
    "scatter_survey_efficiency.png"
)

task_df["start_time"] = task_df["start_time"].dt.tz_localize(None)
task_df["end_time"] = task_df["end_time"].dt.tz_localize(None)


# EXPORT ALL TO ONE EXCEL


output_file = "final_integrated_analysis.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    
    #  task-level（behavior + output）
    task_df.to_excel(writer, sheet_name="task_level", index=False)
    
    # paired behavior/output
    paired_behavior.to_excel(writer, sheet_name="paired_behavior_output", index=False)
    
    # behavior stats
    behavior_results.to_excel(writer, sheet_name="behavior_stats", index=False)
    
    # survey scores
    survey_scores.to_excel(writer, sheet_name="survey_scores", index=False)
    
    # survey stats（t-test）
    survey_results.to_excel(writer, sheet_name="survey_stats", index=False)
    
    # integrated
    integrated.to_excel(writer, sheet_name="integrated_final", index=False)

