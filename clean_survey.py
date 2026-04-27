import pandas as pd


df = pd.read_csv("data/survey.csv", skiprows=[1])


df = df.iloc[1:].reset_index(drop=True)

df["participant_id"] = df.index


print(df.columns)


clean = pd.DataFrame()
clean["participant_id"] = df["participant_id"]

# mapping

clean = pd.DataFrame()
clean["participant_id"] = df["participant_id"]

for col in df.columns:
    if col.endswith("_1"):
        base = col[:-2]
        col_en = base + "_2"

        if col_en in df.columns:
            clean[base + "_L1"] = pd.to_numeric(df[col], errors="coerce")
            clean[base + "_EN"] = pd.to_numeric(df[col_en], errors="coerce")

clean.to_csv("data/clean_survey.csv", index=False)