import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\专利明细情况115313220(仅供UC Berkeley使用)\PT_LCDETAIL.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据（仅读取必要列） ====================
print("正在读取专利明细情况表（仅读取 Symbol, ApplicationDate, PatentTypeCode）...")
# 4.3.1 仅读取必要列，节省内存
df_raw = pd.read_excel(
    INPUT_FILE,
    usecols=['Symbol', 'ApplicationDate', 'PatentTypeCode'],
    dtype={'Symbol': str, 'PatentTypeCode': str}
)
raw_rows = len(df_raw)
print(f"原始读取行数: {raw_rows}")

# ==================== 清洗流程 ====================

# 4.3.2 筛选发明专利（S4901）
df = df_raw[df_raw['PatentTypeCode'] == 'S4901'].copy()
after_patent_filter = len(df)
print(f"筛选发明专利后行数: {after_patent_filter}")

# 4.3.3 提取申请年份
df['Year'] = pd.to_datetime(df['ApplicationDate'], errors='coerce').dt.year
df.dropna(subset=['Year'], inplace=True)
df['Year'] = df['Year'].astype(int)

# 4.3.4 筛选年份范围
df = df[(df['Year'] >= 2008) & (df['Year'] <= 2024)]
after_year_filter = len(df)

# 4.3.5 格式化股票代码
df['Stkcd'] = df['Symbol'].str.zfill(6)

# 4.3.6 按公司-年度计数
df_agg = df.groupby(['Stkcd', 'Year'], as_index=False).size()
df_agg.rename(columns={'size': 'InvPatApply'}, inplace=True)

# 4.3.7 最终保留列
df_final = df_agg[['Stkcd', 'Year', 'InvPatApply']]

# ==================== 统计信息 ====================
final_rows = len(df_final)
unique_stocks = df_final['Stkcd'].nunique()
year_range = f"{df_final['Year'].min()} - {df_final['Year'].max()}"
missing_counts = df_final.isnull().sum()

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("专利明细表（PT_LCDETAIL）清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"原始读取行数: {raw_rows}")
report_lines.append(f"筛选发明专利后行数: {after_patent_filter}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"分组聚合后最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append("\n变量描述统计:")
report_lines.append("InvPatApply 描述统计:")
report_lines.append(df_final['InvPatApply'].describe().to_string())
report_lines.append("\n变量缺失值数量:")
report_lines.append(f"InvPatApply: {missing_counts['InvPatApply']}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "PT_LCDETAIL_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "PT_LCDETAIL_cleaned.csv")
df_final.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "PT_LCDETAIL_sample.xlsx")
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