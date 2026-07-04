import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据\企业新质生产力三级指标表(劳动力、生产工具角度)163242606(仅供北洋大学使用)\NQPF_EnNQPThreeLevelIndLT.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取新质生产力 Excel 文件...")
df_raw = pd.read_excel(INPUT_FILE, dtype={'Symbol': str})
raw_rows = len(df_raw)

# 3.1.1 提取年份（直接使用 EndDate，并转为数值）
df_raw.rename(columns={'EndDate': 'Year'}, inplace=True)
df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce')
df_raw.dropna(subset=['Year'], inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# 3.1.2 筛选年份范围
df = df_raw[(df_raw['Year'] >= 2008) & (df_raw['Year'] <= 2024)].copy()
after_year_filter = len(df)

# 3.1.3 重命名股票代码列
df.rename(columns={'Symbol': 'Stkcd'}, inplace=True)

# 3.1.4 格式化股票代码
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 3.1.5 定义11个指标列名（与抽样文件完全一致）
indicator_cols = [
    'RDPSalaryRatio',
    'RDPersonRatio',
    'HighEduPersonRatio',
    'FixedAssetsRatio',
    'ManufacturCostsRatio',
    'RDPDepAmortRatio',
    'RDPLeaseCostsRatio',
    'DirectInvestment',
    'IntangibleAssetsRatio',
    'AssetTurnover',
    'EquityMultiplierRec'
]

missing_cols = set(indicator_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少指标列: {missing_cols}")

# 指标转为数值
for col in indicator_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3.1.6 缺失值处理：每行缺失个数超过5则删除
df['missing_count'] = df[indicator_cols].isnull().sum(axis=1)
df = df[df['missing_count'] <= 5].copy()
df.drop(columns=['missing_count'], inplace=True)
after_missing_drop = len(df)

# 3.1.7 负值统计
negative_counts = {col: (df[col] < 0).sum() for col in indicator_cols}

# ==================== 熵值法合成 NPro ====================
def entropy_weight_method(df_year, cols):
    X = df_year[cols].copy()
    n = X.shape[0]
    if n < 3:
        return pd.Series([np.nan] * n, index=df_year.index)
    X_filled = X.fillna(X.mean())
    X_norm = (X_filled - X_filled.min()) / (X_filled.max() - X_filled.min() + 1e-12)
    X_norm = X_norm.fillna(0.5).clip(0.001, 0.999)
    P = X_norm / X_norm.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_P = np.log(P)
        ln_P = np.where(P == 0, 0, ln_P)
    E = - (1 / np.log(n)) * (P * ln_P).sum(axis=0)
    d = 1 - E
    W = d / d.sum()
    return (X_norm * W).sum(axis=1)

# 按年分组计算
result_list = []
weights_by_year = {}

for year, group in df.groupby('Year'):
    # 保存原始指标数据用于权重记录
    group_original = group.copy()
    npro_vals = entropy_weight_method(group_original, indicator_cols)
    temp = group_original[['Stkcd', 'Year']].copy()
    temp['NPro'] = npro_vals.values
    result_list.append(temp)

    # 计算并记录权重
    X = group_original[indicator_cols].copy()
    X_filled = X.fillna(X.mean())
    X_norm = (X_filled - X_filled.min()) / (X_filled.max() - X_filled.min() + 1e-12)
    X_norm = X_norm.fillna(0.5).clip(0.001, 0.999)
    P = X_norm / X_norm.sum(axis=0)
    n = X.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_P = np.log(P)
        ln_P = np.where(P == 0, 0, ln_P)
    E = - (1 / np.log(n)) * (P * ln_P).sum(axis=0)
    d = 1 - E
    W = d / d.sum()
    weights_by_year[year] = dict(zip(indicator_cols, W))

df_final = pd.concat(result_list, ignore_index=True)
df_final = df_final.dropna(subset=['NPro'])

# ==================== 统计与报告 ====================
final_rows = len(df_final)
unique_stocks = df_final['Stkcd'].nunique()
year_range = f"{df_final['Year'].min()} - {df_final['Year'].max()}"
yearly_stats = df_final.groupby('Year')['NPro'].agg(['count', 'mean', 'std', 'min', 'max'])
missing_ratios = df[indicator_cols].isnull().mean()

report_lines = [
    "新质生产力表（NQPF_EnNQPThreeLevelIndLT）清洗与指数合成报告",
    "=" * 70,
    f"原始读取行数: {raw_rows}",
    f"筛选年份 (2008-2024) 后行数: {after_year_filter}",
    f"缺失超过5个指标删除后行数: {after_missing_drop}",
    f"最终观测数（公司-年度）: {final_rows}",
    f"涉及公司数: {unique_stocks}",
    f"年份范围: {year_range}",
    "\n--- 各年度 NPro 描述统计 ---",
    yearly_stats.to_string(),
    "\n--- 各指标缺失比例 ---"
]
for col in indicator_cols:
    report_lines.append(f"{col}: {missing_ratios[col]:.2%}")

report_lines.append("\n--- 各指标负值数量 ---")
for col in indicator_cols:
    report_lines.append(f"{col}: {negative_counts[col]}")

report_lines.append("\n--- 各年度熵值法指标权重（部分年份） ---")
sample_years = sorted(weights_by_year.keys())[:5] + sorted(weights_by_year.keys())[-5:]
for year in sample_years:
    if year in weights_by_year:
        report_lines.append(f"\nYear {year}:")
        for col, w in weights_by_year[year].items():
            report_lines.append(f"  {col}: {w:.4f}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "NQPF_EnNQPThreeLevelIndLT_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# 保存清洗后数据
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "NQPF_EnNQPThreeLevelIndLT_cleaned.csv")
df_final.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# 抽样
sample = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "NQPF_EnNQPThreeLevelIndLT_sample.xlsx")
sample.to_excel(sample_path, index=False)

print("\n清洗与指数合成完成！")
print(f"最终观测数: {final_rows} | 涉及公司数: {unique_stocks} | 年份范围: {year_range}")
print(f"报告已保存至: {report_path}")