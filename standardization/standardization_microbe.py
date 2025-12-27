import pandas as pd

# 1. 读取要保留的Bacteria名称
with open("/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/bacteria_wang_names0724.txt", "r", encoding="utf-8") as f:
    keep_names = set(line.strip() for line in f if line.strip())

# 2. 读取五元组文件
df = pd.read_csv("quintuple0724.csv", sep=",", dtype=str)

# 3. 过滤
def bacteria_in_keep(row):
    if row["Subject_type"] == "Bacteria" and row["Subject_name"] not in keep_names:
        return False
    if row["Object_type"] == "Bacteria" and row["Object_name"] not in keep_names:
        return False
    return True

df_filtered = df[df.apply(bacteria_in_keep, axis=1)]

# 4. 保存
df_filtered.to_csv("final_use0724.tsv", sep="\t", index=False)
print("✅ 已保存筛选后的五元组文件：final_use0724.tsv")