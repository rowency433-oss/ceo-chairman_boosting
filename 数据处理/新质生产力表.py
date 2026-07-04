import pandas as pd
import os
import shutil

# 配置路径
INPUT_FILE = r"D:\how dare you\2026统计建模\数据处理\第一次处理\NQPF_EnNQPThreeLevelIndLT_cleaned.csv"
OUTPUT_DIR_TABLES = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表的报告与抽样"

os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# 目标文件路径
TARGET_FILE = os.path.join(OUTPUT_DIR_TABLES, "df_npro.csv")
SAMPLE_FILE = os.path.join(OUTPUT_DIR_REPORT, "df_npro_sample.xlsx")

# 复制重命名
shutil.copy2(INPUT_FILE, TARGET_FILE)
print(f"已复制并重命名为: {TARGET_FILE}")

# 读取并抽样
df = pd.read_csv(TARGET_FILE, dtype={'Stkcd': str})
sample = df.sample(n=min(10, len(df)), random_state=42)
sample.to_excel(SAMPLE_FILE, index=False)
print(f"抽样文件已保存: {SAMPLE_FILE}")

# 简要统计
print(f"观测数: {len(df)}, 公司数: {df['Stkcd'].nunique()}, 年份范围: {df['Year'].min()} - {df['Year'].max()}")