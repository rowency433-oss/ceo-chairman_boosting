import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\上市公司基本信息163004282(仅供北洋大学使用)\CSp_ListedCoInfoAnl.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取公司基本信息 Excel 文件...")
df_raw = pd.read_excel(INPUT_FILE, dtype={'Symbol': str})
raw_rows = len(df_raw)
print(f"原始读取行数: {raw_rows}")

# ==================== 清洗流程 ====================

# 2.4.1 保留必要字段
required_cols = ['Symbol', 'EstablishDate', 'ListingDate', 'IndustryName', 'IndustryCode', 'ListingState']
missing_cols = set(required_cols) - set(df_raw.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df_raw[required_cols].copy()

# 2.4.2 重命名列
df.rename(columns={'Symbol': 'Stkcd'}, inplace=True)

# 2.4.3 格式化股票代码（补零至6位）
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 2.4.4 提取成立年份（容错处理，将非日期转为 NaT 再提取年份）
df['EstablishDate'] = pd.to_datetime(df['EstablishDate'], errors='coerce')
df['EstYear'] = df['EstablishDate'].dt.year

# 2.4.5 提取上市年份
df['ListingDate'] = pd.to_datetime(df['ListingDate'], errors='coerce')
df['ListYear'] = df['ListingDate'].dt.year

# 2.4.6 标记ST状态
# 转为字符串，忽略大小写判断是否包含 'ST' 或 'PT'
listing_state_str = df['ListingState'].fillna('').astype(str).str.upper()
df['IsST'] = listing_state_str.str.contains('ST|PT').astype(int)

# 2.4.7 标记金融/房地产
# 提取 IndustryCode 的首字母，若缺失则填充为空字符串
ind_code_first = df['IndustryCode'].fillna('').astype(str).str[0]
df['IsFinReal'] = ind_code_first.isin(['J', 'K']).astype(int)

# 2.4.8 最终保留列
final_cols = ['Stkcd', 'EstYear', 'ListYear', 'IndustryCode', 'IndustryName', 'IsST', 'IsFinReal']
df = df[final_cols]

# ==================== 统计信息 ====================
final_rows = len(df)
st_count = df['IsST'].sum()
st_ratio = st_count / final_rows if final_rows > 0 else 0
finreal_count = df['IsFinReal'].sum()
finreal_ratio = finreal_count / final_rows if final_rows > 0 else 0

# 缺失值统计
missing_est = df['EstYear'].isnull().sum()
missing_list = df['ListYear'].isnull().sum()
missing_ind_code = df['IndustryCode'].isnull().sum()
missing_ind_name = df['IndustryName'].isnull().sum()

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("公司基本信息表 清洗报告")
report_lines.append("=" * 50)
report_lines.append(f"原始读取行数: {raw_rows}")
report_lines.append(f"最终观测数（公司数）: {final_rows}")
report_lines.append(f"标记为ST/*ST/PT的公司数: {st_count}，占比: {st_ratio:.2%}")
report_lines.append(f"标记为金融/房地产的公司数: {finreal_count}，占比: {finreal_ratio:.2%}")
report_lines.append("\n变量描述统计:")
report_lines.append("EstYear 描述统计:")
report_lines.append(df['EstYear'].describe().to_string())
report_lines.append("\nListYear 描述统计:")
report_lines.append(df['ListYear'].describe().to_string())
report_lines.append("\n各变量缺失值数量:")
report_lines.append(f"EstYear 缺失: {missing_est}")
report_lines.append(f"ListYear 缺失: {missing_list}")
report_lines.append(f"IndustryCode 缺失: {missing_ind_code}")
report_lines.append(f"IndustryName 缺失: {missing_ind_name}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "CSp_ListedCoInfoAnl_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "CSp_ListedCoInfoAnl_cleaned.csv")
df.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "CSp_ListedCoInfoAnl_sample.xlsx")
sample.to_excel(sample_path, index=False)

# ==================== 控制台输出摘要 ====================
print("\n清洗完成！")
print(f"清洗后数据已保存至: {cleaned_path}")
print(f"报告已保存至: {report_path}")
print(f"抽样数据已保存至: {sample_path}")
print("\n--- 报告摘要 ---")
print(f"最终公司数: {final_rows}")
print(f"ST 公司数: {st_count} ({st_ratio:.2%})")
print(f"金融/房地产公司数: {finreal_count} ({finreal_ratio:.2%})")