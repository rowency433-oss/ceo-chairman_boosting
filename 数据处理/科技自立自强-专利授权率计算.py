import pandas as pd
import numpy as np
import os

# ==================== 配置参数 ====================
INPUT_DIR = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_TABLES = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表的报告与抽样"

# 输入文件路径
RD_FILE = os.path.join(INPUT_DIR, "PT_LCRDSPENDING_cleaned.csv")
GRANT_FILE = os.path.join(INPUT_DIR, "TIRD_EntRDTecCap_cleaned.csv")
APPLY_FILE = os.path.join(INPUT_DIR, "PT_LCDETAIL_cleaned.csv")
VALID_FILE = os.path.join(INPUT_DIR, "PT_VALID_cleaned.csv")

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取输入文件...")
df_rd = pd.read_csv(RD_FILE, dtype={'Stkcd': str})
df_grant = pd.read_csv(GRANT_FILE, dtype={'Stkcd': str})
df_apply = pd.read_csv(APPLY_FILE, dtype={'Stkcd': str})
df_valid = pd.read_csv(VALID_FILE, dtype={'Stkcd': str})

print(f"研发投入表行数: {len(df_rd)}")
print(f"发明专利授权表行数: {len(df_grant)}")
print(f"发明专利申请表行数: {len(df_apply)}")
print(f"有效发明专利存量表行数: {len(df_valid)}")

# 统一年份类型
for df in [df_rd, df_grant, df_apply, df_valid]:
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)

# ==================== 步骤 4.4：计算发明专利授权率 ====================
print("\n正在计算发明专利授权率...")
df_patent = df_grant.merge(
    df_apply[['Stkcd', 'Year', 'InvPatApply']],
    on=['Stkcd', 'Year'],
    how='left'
)
# 缺失的申请数视为0
df_patent['InvPatApply'] = df_patent['InvPatApply'].fillna(0)
# 计算授权率（申请数为0时授权率为0）
df_patent['InvPatGrantRatio'] = np.where(
    df_patent['InvPatApply'] == 0,
    0,
    df_patent['InvPatGrant'] / df_patent['InvPatApply']
)
# 保留必要列
df_patent_combined = df_patent[['Stkcd', 'Year', 'InvPatGrant', 'LnInvPatGrant',
                                'InvPatApply', 'InvPatGrantRatio']].copy()

# ==================== 步骤 4.6：合并四张子表 ====================
print("正在合并研发投入、专利及存量表...")
# 以研发投入表为基准
df_merged = df_rd.merge(
    df_patent_combined,
    on=['Stkcd', 'Year'],
    how='left'
).merge(
    df_valid[['Stkcd', 'Year', 'ValidInvPat', 'LnValidInvPat']],
    on=['Stkcd', 'Year'],
    how='left'
)

# 定义五个合成指标
indicators = [
    'RDRatio',
    'RDPersonRatio',
    'LnInvPatGrant',
    'InvPatGrantRatio',
    'LnValidInvPat'
]

# 检查指标列是否存在，缺失则填充NaN并警告
for col in indicators:
    if col not in df_merged.columns:
        print(f"警告：指标列 '{col}' 不存在，将创建并填充 NaN")
        df_merged[col] = np.nan

# ==================== 熵值法合成 STSelf 指数 ====================
def entropy_weight_method(df_year, cols):
    """
    对某一年的数据执行熵值法，返回该年每个样本的 STSelf 值
    """
    X = df_year[cols].copy()
    n = X.shape[0]
    if n < 3:
        return pd.Series([np.nan] * n, index=df_year.index), None

    # 缺失值填补：用当年均值填补
    X_filled = X.fillna(X.mean())
    # 若某列全为NaN（均值仍为NaN），则用0填充（作为最后手段）
    X_filled = X_filled.fillna(0)

    # Min-Max 标准化至 [0.001, 0.999]
    X_min = X_filled.min()
    X_max = X_filled.max()
    X_norm = (X_filled - X_min) / (X_max - X_min + 1e-12)
    X_norm = X_norm.fillna(0.5).clip(0.001, 0.999)

    # 计算比重矩阵 P
    col_sums = X_norm.sum(axis=0)
    P = X_norm / col_sums

    # 计算信息熵 E
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_P = np.log(P)
        ln_P = np.where(P == 0, 0, ln_P)
    E = - (1 / np.log(n)) * (P * ln_P).sum(axis=0)

    # 计算权重 W
    d = 1 - E
    if d.sum() == 0:
        # 所有指标信息熵均为0，采用等权重
        W = pd.Series(1/len(cols), index=cols)
    else:
        W = d / d.sum()

    # 综合指数
    STSelf = (X_norm * W).sum(axis=1)
    return STSelf, W

print("\n正在按年度合成 STSelf 指数...")
result_list = []
weights_by_year = {}

for year, group in df_merged.groupby('Year'):
    stself_series, weights = entropy_weight_method(group, indicators)
    group_result = group[['Stkcd', 'Year']].copy()
    group_result['STSelf'] = stself_series.values
    result_list.append(group_result)
    if weights is not None:
        weights_by_year[year] = weights.to_dict()

df_stself = pd.concat(result_list, ignore_index=True)
df_stself = df_stself.dropna(subset=['STSelf'])  # 移除样本不足年份

# ==================== 统计信息 ====================
# 专利授权率描述
patent_ratio_desc = df_patent_combined['InvPatGrantRatio'].describe()

# 合并后观测数
merged_rows = len(df_merged)
unique_stocks = df_merged['Stkcd'].nunique()
year_range = f"{df_merged['Year'].min()} - {df_merged['Year'].max()}"

# 各年度 STSelf 描述统计
yearly_stats = df_stself.groupby('Year')['STSelf'].agg(['count', 'mean', 'std', 'min', 'max'])

# 五个指标缺失比例
missing_ratios = df_merged[indicators].isnull().mean()

# ==================== 生成报告 ====================
print("\n正在生成报告...")
report_lines = []
report_lines.append("科技自立自强表构造报告")
report_lines.append("=" * 60)
report_lines.append("【输入表行数】")
report_lines.append(f"研发投入表: {len(df_rd)}")
report_lines.append(f"发明专利授权表: {len(df_grant)}")
report_lines.append(f"发明专利申请表: {len(df_apply)}")
report_lines.append(f"有效发明专利存量表: {len(df_valid)}")
report_lines.append("")

report_lines.append("【专利授权率计算】")
report_lines.append(f"专利合并表观测数: {len(df_patent_combined)}")
report_lines.append("InvPatGrantRatio 描述统计:")
report_lines.append(patent_ratio_desc.to_string())
report_lines.append("")

report_lines.append("【合并后数据概况】")
report_lines.append(f"合并后观测数: {merged_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"年份范围: {year_range}")
report_lines.append("")

report_lines.append("【五个指标缺失比例】")
for col in indicators:
    report_lines.append(f"{col}: {missing_ratios[col]:.2%}")
report_lines.append("")

report_lines.append("【各年度 STSelf 描述统计】")
report_lines.append(yearly_stats.to_string())
report_lines.append("")

report_lines.append("【熵值法权重（部分年份）】")
sample_years = sorted(weights_by_year.keys())[:3] + sorted(weights_by_year.keys())[-3:]
sample_years = sorted(set(sample_years))
for year in sample_years:
    if year in weights_by_year:
        report_lines.append(f"\nYear {year}:")
        for col, w in weights_by_year[year].items():
            report_lines.append(f"  {col}: {w:.4f}")
report_lines.append("")

report_lines.append("【最终 STSelf 缺失值数量】")
report_lines.append(f"STSelf 缺失数: {df_stself['STSelf'].isnull().sum()}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "stself_construction_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存数据表 ====================
print("正在保存输出文件...")
patent_combined_path = os.path.join(OUTPUT_DIR_TABLES, "df_patent_combined.csv")
df_patent_combined.to_csv(patent_combined_path, index=False, encoding='utf-8-sig')

stself_path = os.path.join(OUTPUT_DIR_TABLES, "df_stself.csv")
df_stself.to_csv(stself_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample_patent = df_patent_combined.sample(n=min(SAMPLE_SIZE, len(df_patent_combined)), random_state=SEED)
sample_patent_path = os.path.join(OUTPUT_DIR_REPORT, "df_patent_combined_sample.xlsx")
sample_patent.to_excel(sample_patent_path, index=False)

sample_stself = df_stself.sample(n=min(SAMPLE_SIZE, len(df_stself)), random_state=SEED)
sample_stself_path = os.path.join(OUTPUT_DIR_REPORT, "df_stself_sample.xlsx")
sample_stself.to_excel(sample_stself_path, index=False)

# ==================== 控制台摘要 ====================
print("\n科技自立自强表构造完成！")
print(f"专利合并表保存至: {patent_combined_path}")
print(f"STSelf 指数表保存至: {stself_path}")
print(f"报告保存至: {report_path}")
print("\n--- 快速摘要 ---")
print(f"合并后观测数: {merged_rows}")
print(f"涉及公司数: {unique_stocks}")
print(f"年份范围: {year_range}")
print(f"STSelf 有效观测数: {len(df_stself)}")