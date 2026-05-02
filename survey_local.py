import sqlite3
import pandas as pd

# 连接数据库
conn = sqlite3.connect("survey_local.db")

# 查看所有表
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

# 读取某个表（比如 survey）
df = pd.read_sql_query("SELECT * FROM messages", conn)
df.to_csv("output.csv", index=False)

df2 = pd.read_sql_query("SELECT * FROM participants", conn)
df2.to_csv("output_participants.csv", index=False)

print(df.head(20))
