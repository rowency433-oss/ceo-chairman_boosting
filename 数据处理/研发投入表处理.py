import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\研发投入情况表115449623(仅供UC Berkeley使用)\PT_LCRDSPENDING.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取研发投入情况表...")
df_raw = pd.read_excel(INPUT_FILE, dtype={'Symbol': str})
raw_rows = len(df_raw)

# ==================== 清洗流程 ====================

# 4.1.1 提取年份
df_raw['Year'] = pd.to_datetime(df_raw['EndDate'], errors='coerce').dt.year
df_raw.dropna(subset=['Year'], inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# 4.1.2 筛选年份范围
df = df_raw[(df_raw['Year'] >= 2008) & (df_raw['Year'] <= 2024)].copy()
after_year_filter = len(df)

# 4.1.3 筛选合并报表（若字段存在）
if 'StateTypeCode' in df.columns:
    df['StateTypeCode'] = pd.to_numeric(df['StateTypeCode'], errors='coerce')
    # 仅保留合并报表（值为1）或缺失的记录
    df = df[(df['StateTypeCode'] == 1) | (df['StateTypeCode'].isna())]
after_state_filter = len(df)

# 4.1.4 保留必要字段
required_cols = ['Symbol', 'Year', 'RDSpendSumRatio', 'RDPersonRatio']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# 4.1.5 重命名列
df.rename(columns={
    'Symbol': 'Stkcd',
    'RDSpendSumRatio': 'RDRatio'
}, inplace=True)

# 4.1.6 格式化股票代码
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 4.1.7 比例转为小数
# 先将两列转为数值类型
df['RDRatio'] = pd.to_numeric(df['RDRatio'], errors='coerce')
df['RDPersonRatio'] = pd.to_numeric(df['RDPersonRatio'], errors='coerce')

# 检查最大值是否大于1，若是则除以100
if df['RDRatio'].max() > 1 or df['RDPersonRatio'].max() > 1:
    print("检测到比例为百分比形式（最大值 > 1），执行除以100操作。")
    df['RDRatio'] = df['RDRatio'] / 100
    df['RDPersonRatio'] = df['RDPersonRatio'] / 100
else:
    print("比例已为小数形式，无需转换。")

# 4.1.8 最终保留列
final_cols = ['Stkcd', 'Year', 'RDRatio', 'RDPersonRatio']
df_final = df[final_cols]

# ==================== 统计信息 ====================
final_rows = len(df_final)
unique_stocks = df_final['Stkcd'].nunique()
year_range = f"{df_final['Year'].min()} - {df_final['Year'].max()}"
missing_counts = df_final.isnull().sum()

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("研发投入情况表（PT_LCRDSPENDING）清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"原始读取行数: {raw_rows}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"筛选合并报表后行数: {after_state_filter}")
report_lines.append(f"最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append("\n变量描述统计（比例已转为小数）:")
report_lines.append("RDRatio 描述统计:")
report_lines.append(df_final['RDRatio'].describe().to_string())
report_lines.append("\nRDPersonRatio 描述统计:")
report_lines.append(df_final['RDPersonRatio'].describe().to_string())
report_lines.append("\n变量缺失值数量:")
report_lines.append(f"RDRatio: {missing_counts['RDRatio']}")
report_lines.append(f"RDPersonRatio: {missing_counts['RDPersonRatio']}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "PT_LCRDSPENDING_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "PT_LCRDSPENDING_cleaned.csv")
df_final.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "PT_LCRDSPENDING_sample.xlsx")
sample.to_excel(sample_path, index=False)

# ==================== 控制台输出摘要 ====================
print("\n清洗完成！")
print(f"清洗后数据已保存至: {cleaned_path}")
print(f"报告已保存至: {report_path}")
print(f"抽样数据已保存至: {sample_path}")
print("\n--- 报告摘要 ---")
print(f"最终观测数: {final_rows}")
print(f"涉及公司数: {unique_stocks}")
print(f"年份范围: {year_range}")