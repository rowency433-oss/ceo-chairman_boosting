import pandas as pd
import os

# 文件路径
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\公司基本信息\公司基本信息.xlsx"
OUTPUT_DIR = r"D:\how dare you\2026统计建模\数据处理\原始数据抽样"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "公司基本信息_sample.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 仅读取指定四列
usecols = ['Scode', 'Estbdt', 'Listdt', 'IndcodeB']
df_raw = pd.read_excel(INPUT_FILE, usecols=usecols, dtype={'Scode': str})

print(f"原始读取行数: {len(df_raw)}")

# 随机抽样 20 行
SAMPLE_SIZE = min(20, len(df_raw))
df_sample = df_raw.sample(n=SAMPLE_SIZE, random_state=42)

# 保存原始抽样数据
df_sample.to_excel(OUTPUT_FILE, index=False)

print(f"抽样完成，保存至: {OUTPUT_FILE}")