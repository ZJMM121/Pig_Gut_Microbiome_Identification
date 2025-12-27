import pandas as pd
import os

# 读取五元组文件
df = pd.read_csv("quintuple0724.csv", sep=",")

# 创建输出目录
os.makedirs("entity_names0724", exist_ok=True)

# 统计所有实体类型
entity_types = set(df["Subject_type"]).union(set(df["Object_type"]))

for etype in entity_types:
    # 获取所有该类型的实体名（主语和宾语）
    subject_names = df.loc[df["Subject_type"] == etype, "Subject_name"]
    object_names = df.loc[df["Object_type"] == etype, "Object_name"]
    all_names = pd.Series(list(subject_names) + list(object_names)).drop_duplicates().sort_values()
    # 保存到文件
    out_path = f"entity_names0724/{etype}_names.txt"
    all_names.to_csv(out_path, index=False, header=False)
    print(f"已保存: {out_path} ({len(all_names)} 个)")
