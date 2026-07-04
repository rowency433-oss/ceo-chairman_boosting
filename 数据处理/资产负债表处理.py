import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告（不影响数据正确性）
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\资产负债表122033492(仅供UC Berkeley使用)\FS_Combas.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取 Excel 文件...")
df_raw = pd.read_excel(INPUT_FILE, dtype={'Stkcd': str})
raw_rows = len(df_raw)

# ==================== 清洗流程 ====================

# 2.1.1 筛选合并报表
df = df_raw[df_raw['Typrep'] == 'A'].copy()
after_typrep = len(df)

# 2.1.2 提取年份
df['Year'] = pd.to_datetime(df['Accper']).dt.year

# 2.1.3 筛选年份范围
df = df[(df['Year'] >= 2008) & (df['Year'] <= 2024)]
after_year_filter = len(df)

# 2.1.4 保留必要字段
required_cols = ['Stkcd', 'Year', 'A001000000', 'A002000000', 'A001212000']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# 2.1.5 重命名列
df.rename(columns={
    'A001000000': 'TotalAssets',
    'A002000000': 'TotalLiabilities',
    'A001212000': 'FixedAssets'
}, inplace=True)

# 2.1.6 格式化股票代码（补零至6位）
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 2.1.7 强制转换为数值类型，无法转换的变为 NaN，然后删除缺失值
for col in ['TotalAssets', 'TotalLiabilities', 'FixedAssets']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=['TotalAssets', 'TotalLiabilities', 'FixedAssets'], inplace=True)

# 2.1.8 异常值处理（总资产 <= 0 的行删除）
df = df[df['TotalAssets'] > 0]

# 2.1.9 计算衍生变量
df['Size'] = np.log(df['TotalAssets'])
df['Leverage'] = df['TotalLiabilities'] / df['TotalAssets']
df['PPE'] = df['FixedAssets'] / df['TotalAssets']

# 2.1.10 最终保留列
final_cols = ['Stkcd', 'Year', 'Size', 'Leverage', 'PPE']
df = df[final_cols]

# ==================== 统计信息 ====================
final_rows = len(df)
unique_stocks = df['Stkcd'].nunique()
year_range = f"{df['Year'].min()} - {df['Year'].max()}"

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("资产负债表 (FS_Combas) 清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"原始读取行数: {raw_rows}")
report_lines.append(f"筛选 Typrep='A' 后行数: {after_typrep}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append("\n变量描述统计:")
report_lines.append(df[['Size', 'Leverage', 'PPE']].describe().to_string())
report_lines.append("\n变量缺失值数量:")
report_lines.append(df[['Size', 'Leverage', 'PPE']].isnull().sum().to_string())

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "FS_Combas_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "FS_Combas_cleaned.csv")
df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "FS_Combas_sample.xlsx")
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