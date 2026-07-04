import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE_1 = r"D:\how dare you\2026统计建模\数据\大股东持股\大股东持股_1.xlsx"
INPUT_FILE_2 = r"D:\how dare you\2026统计建模\数据\大股东持股\大股东持股_2.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取并合并两个文件 ====================
print("正在读取大股东持股_1.xlsx...")
df1 = pd.read_excel(INPUT_FILE_1, dtype={'SCode': str})
rows1 = len(df1)
print(f"文件1行数: {rows1}")

print("正在读取大股东持股_2.xlsx...")
df2 = pd.read_excel(INPUT_FILE_2, dtype={'SCode': str})
rows2 = len(df2)
print(f"文件2行数: {rows2}")

# 纵向拼接
df_raw = pd.concat([df1, df2], ignore_index=True)
raw_rows = len(df_raw)
print(f"合并后总行数: {raw_rows}")

# ==================== 清洗流程 ====================

# 2.3.2 提取年份（容错处理）
df_raw['Year'] = pd.to_datetime(df_raw['Date'], errors='coerce').dt.year
df_raw.dropna(subset=['Year'], inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# 2.3.3 筛选年份范围
df = df_raw[(df_raw['Year'] >= 2008) & (df_raw['Year'] <= 2024)].copy()
after_year_filter = len(df)

# 2.3.4 筛选前十大股东
# 确保 Num 列为数值类型
df['Num'] = pd.to_numeric(df['Num'], errors='coerce')
df.dropna(subset=['Num'], inplace=True)
df = df[df['Num'] <= 10]
after_num_filter = len(df)

# 2.3.5 保留必要字段
required_cols = ['SCode', 'Year', 'ShrRt']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# 2.3.6 重命名列
df.rename(columns={'SCode': 'Stkcd'}, inplace=True)

# 2.3.7 格式化股票代码（补零至6位）
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 2.3.8 持股比例转为小数
# 先转为数值类型，无法转换的变为 NaN
df['ShrRt'] = pd.to_numeric(df['ShrRt'], errors='coerce')
df.dropna(subset=['ShrRt'], inplace=True)

# 检查最大值是否大于1，若是则除以100
if df['ShrRt'].max() > 1:
    print("检测到 ShrRt 为百分比形式（最大值 > 1），执行除以100操作。")
    df['ShrRt'] = df['ShrRt'] / 100
else:
    print("ShrRt 已为小数形式，无需转换。")

# 2.3.9 按公司-年度求和
df_grouped = df.groupby(['Stkcd', 'Year'], as_index=False)['ShrRt'].sum()
df_grouped.rename(columns={'ShrRt': 'Top10HoldRatio'}, inplace=True)

# 2.3.10 最终保留列
df = df_grouped[['Stkcd', 'Year', 'Top10HoldRatio']]

# ==================== 统计信息 ====================
final_rows = len(df)
unique_stocks = df['Stkcd'].nunique()
year_range = f"{df['Year'].min()} - {df['Year'].max()}"
missing_count = df['Top10HoldRatio'].isnull().sum()  # 标量

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("大股东持股表 清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"文件1原始行数: {rows1}")
report_lines.append(f"文件2原始行数: {rows2}")
report_lines.append(f"合并后总行数: {raw_rows}")
report_lines.append(f"筛选 Num <= 10 后行数: {after_num_filter}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"分组聚合后最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append("\n变量描述统计（Top10HoldRatio 已为小数）:")
report_lines.append(df['Top10HoldRatio'].describe().to_string())
report_lines.append(f"\n变量缺失值数量: {missing_count}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "大股东持股_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "大股东持股_cleaned.csv")
df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "大股东持股_sample.xlsx")
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