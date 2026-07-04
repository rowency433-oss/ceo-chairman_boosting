import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\专利有效情况表(1990-2017)121327553(仅供UC Berkeley使用)\PT_VALID.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据（不指定 ClassifySign 类型） ====================
print("正在读取有效发明专利存量表...")
df_raw = pd.read_excel(
    INPUT_FILE,
    dtype={'Symbol': str, 'MergeSign': str}  # ClassifySign 不指定，让其推断
)
raw_rows = len(df_raw)

# ==================== 清洗流程 ====================

# 4.5.1 提取年份
df_raw['Year'] = pd.to_datetime(df_raw['EndDate'], errors='coerce').dt.year
df_raw.dropna(subset=['Year'], inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# 4.5.2 筛选年份范围
df = df_raw[(df_raw['Year'] >= 2008) & (df_raw['Year'] <= 2024)].copy()
after_year_filter = len(df)

# 4.5.3 筛选上市公司本身小计 (ClassifySign == 4)
# 先将 ClassifySign 转为数值，无法转换的变为 NaN，然后筛选值为 4 的行
df['ClassifySign'] = pd.to_numeric(df['ClassifySign'], errors='coerce')
df = df[df['ClassifySign'] == 4]
after_classify_filter = len(df)

# 4.5.4 筛选合并统计值 (MergeSign == 'Y')
# 确保 MergeSign 为字符串，并剔除缺失
df['MergeSign'] = df['MergeSign'].fillna('').astype(str)
df = df[df['MergeSign'] == 'Y']
after_merge_filter = len(df)

# 4.5.5 保留必要字段
required_cols = ['Symbol', 'Year', 'Invention']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# 4.5.6 重命名列
df.rename(columns={
    'Symbol': 'Stkcd',
    'Invention': 'ValidInvPat'
}, inplace=True)

# 4.5.7 格式化股票代码
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 4.5.8 缺失值处理：记录原始缺失数量，然后填充为0
original_missing = df['ValidInvPat'].isnull().sum()
df['ValidInvPat'] = pd.to_numeric(df['ValidInvPat'], errors='coerce')
df['ValidInvPat'] = df['ValidInvPat'].fillna(0)

# 4.5.9 计算对数
df['LnValidInvPat'] = np.log1p(df['ValidInvPat'])

# 4.5.10 最终保留列
final_cols = ['Stkcd', 'Year', 'ValidInvPat', 'LnValidInvPat']
df_final = df[final_cols]

# ==================== 统计信息 ====================
final_rows = len(df_final)
unique_stocks = df_final['Stkcd'].nunique()
year_range = f"{df_final['Year'].min()} - {df_final['Year'].max()}"
missing_counts = df_final.isnull().sum()

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("有效发明专利存量表（PT_VALID）清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"原始读取行数: {raw_rows}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"筛选 ClassifySign=4 且 MergeSign='Y' 后行数: {after_merge_filter}")
report_lines.append(f"最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append(f"原始缺失值数量（填充前）: {original_missing}")
report_lines.append("\n变量描述统计:")
report_lines.append("ValidInvPat 描述统计:")
report_lines.append(df_final['ValidInvPat'].describe().to_string())
report_lines.append("\nLnValidInvPat 描述统计:")
report_lines.append(df_final['LnValidInvPat'].describe().to_string())
report_lines.append("\n变量缺失值数量（填充后）:")
report_lines.append(f"ValidInvPat: {missing_counts['ValidInvPat']}")
report_lines.append(f"LnValidInvPat: {missing_counts['LnValidInvPat']}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "PT_VALID_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "PT_VALID_cleaned.csv")
df_final.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "PT_VALID_sample.xlsx")
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
print(f"原始缺失值数量: {original_missing}")