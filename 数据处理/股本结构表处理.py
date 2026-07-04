import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\股本结构\股本结构.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据（基于真实表头） ====================
print("正在读取股本结构 Excel 文件...")
# 真实列名：SCode, Date, TotlShr, RShr_StOw, RShr_SrMgmt
df_raw = pd.read_excel(INPUT_FILE, dtype={'SCode': str})
raw_rows = len(df_raw)

# 打印实际列名以供核对
print("实际列名:", df_raw.columns.tolist())

# ==================== 清洗流程 ====================

# 2.2.1 提取年份（增加容错，无法解析转为 NaT）
df_raw['Year'] = pd.to_datetime(df_raw['Date'], errors='coerce').dt.year

# 删除年份缺失的行（如无效日期或表头残留）
df_raw.dropna(subset=['Year'], inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# 2.2.2 筛选年份范围
df = df_raw[(df_raw['Year'] >= 2008) & (df_raw['Year'] <= 2024)].copy()
after_year_filter = len(df)

# 2.2.3 保留必要字段（直接使用原始中文列名）
required_cols = ['SCode', 'Year', 'TotlShr', 'RShr_StOw']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# 2.2.4 重命名列（按清洗后命名规范）
df.rename(columns={
    'SCode': 'Stkcd',
    'TotlShr': 'TotalShares',
    'RShr_StOw': 'StateOwnedShares'
}, inplace=True)

# 2.2.5 格式化股票代码（补零至6位）
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 2.2.6 国有股缺失处理（填充为0）
df['StateOwnedShares'] = df['StateOwnedShares'].fillna(0)

# 2.2.7 总股本数值转换 + 异常处理
df['TotalShares'] = pd.to_numeric(df['TotalShares'], errors='coerce')
df['StateOwnedShares'] = pd.to_numeric(df['StateOwnedShares'], errors='coerce').fillna(0)

# 删除总股本缺失或 ≤0 的行
df.dropna(subset=['TotalShares'], inplace=True)
df = df[df['TotalShares'] > 0]

# 2.2.8 计算国有股比例
df['StateShare'] = df['StateOwnedShares'] / df['TotalShares']

# 2.2.9 最终保留列
final_cols = ['Stkcd', 'Year', 'TotalShares', 'StateShare']
df = df[final_cols]

# ==================== 统计信息 ====================
final_rows = len(df)
unique_stocks = df['Stkcd'].nunique()
year_range = f"{df['Year'].min()} - {df['Year'].max()}"

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("股本结构表 清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"原始读取行数: {raw_rows}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"删除总股本异常后行数: {final_rows}")
report_lines.append(f"最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append("\n变量描述统计:")
report_lines.append(df[['TotalShares', 'StateShare']].describe().to_string())
report_lines.append("\n变量缺失值数量:")
report_lines.append(df[['TotalShares', 'StateShare']].isnull().sum().to_string())

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "股本结构_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "股本结构_cleaned.csv")
df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "股本结构_sample.xlsx")
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