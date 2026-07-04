import pandas as pd
import numpy as np
import os
from scipy.stats.mstats import winsorize

# ==================== 配置参数 ====================
INPUT_DIR_TABLES = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表"
OUTPUT_DIR_FINAL = r"D:\how dare you\2026统计建模\数据处理\第三次处理-最终数据"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第三次处理-最终数据\报告与抽样"

CONTROL_FILE = os.path.join(INPUT_DIR_TABLES, "df_control.csv")
NPRO_FILE = os.path.join(INPUT_DIR_TABLES, "df_npro.csv")
STSELF_FILE = os.path.join(INPUT_DIR_TABLES, "df_stself.csv")
EXECUTIVE_FILE = os.path.join(INPUT_DIR_TABLES, "df_executive.csv")

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_FINAL, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 1. 读取数据 ====================
print("正在读取四张构造表...")
df_control = pd.read_csv(CONTROL_FILE, dtype={'Stkcd': str, 'IndustryCode': str})
df_npro = pd.read_csv(NPRO_FILE, dtype={'Stkcd': str})
df_stself = pd.read_csv(STSELF_FILE, dtype={'Stkcd': str})
df_executive = pd.read_csv(EXECUTIVE_FILE, dtype={'Stkcd': str})

for df in [df_control, df_npro, df_stself, df_executive]:
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)

print(f"控制变量表: {len(df_control)}")
print(f"新质生产力表: {len(df_npro)}")
print(f"科技自立自强表: {len(df_stself)}")
print(f"高管特征表: {len(df_executive)}")

# ==================== 2. 四表合并 ====================
print("\n开始四表合并...")
step_counts = {'控制变量表（基准）': len(df_control)}
df = df_control.copy()

df = df.merge(df_npro[['Stkcd', 'Year', 'NPro']], on=['Stkcd', 'Year'], how='left')
step_counts['连接 NPro 后'] = len(df)

df = df.merge(df_stself[['Stkcd', 'Year', 'STSelf']], on=['Stkcd', 'Year'], how='left')
step_counts['连接 STSelf 后'] = len(df)

df = df.merge(df_executive, on=['Stkcd', 'Year'], how='left')
step_counts['连接高管特征后'] = len(df)

# ==================== 3. 计算高管持股比例 ====================
print("计算高管持股比例...")
if 'TotalShares' not in df.columns:
    raise KeyError("缺少 TotalShares 字段")

df['ShareRatio_CEO'] = np.where(
    df['TotalShares'].fillna(0) > 0,
    df['SharEnd_CEO'] / df['TotalShares'],
    np.nan
)
df['ShareRatio_Chair'] = np.where(
    df['TotalShares'].fillna(0) > 0,
    df['SharEnd_Chair'] / df['TotalShares'],
    np.nan
)
df.drop(columns=['SharEnd_CEO', 'SharEnd_Chair'], inplace=True, errors='ignore')

# ==================== 4. 全局缩尾处理 ====================
print("执行全局缩尾 (1% 和 99% 分位数)...")
winsor_cols = [
    'Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
    'BoardSize', 'IndepRatio', 'LnFirmAge',
    'NPro', 'STSelf',
    'LnAge_CEO', 'ShareRatio_CEO', 'LnAge_Chair', 'ShareRatio_Chair'
]

desc_before = df[winsor_cols].describe().T

for col in winsor_cols:
    if col not in df.columns:
        continue
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)

desc_after = df[winsor_cols].describe().T

winsor_compare = pd.DataFrame({
    '变量': winsor_cols,
    '缩尾前最小值': desc_before['min'].values,
    '缩尾后最小值': desc_after['min'].values,
    '缩尾前最大值': desc_before['max'].values,
    '缩尾后最大值': desc_after['max'].values,
    '缩尾前均值': desc_before['mean'].values,
    '缩尾后均值': desc_after['mean'].values,
    '缩尾前标准差': desc_before['std'].values,
    '缩尾后标准差': desc_after['std'].values
})
winsor_path = os.path.join(OUTPUT_DIR_REPORT, "winsorization_summary.xlsx")
winsor_compare.to_excel(winsor_path, index=False)

# ==================== 5. 缺失值最终处理 ====================
print("\n处理缺失值...")
missing_log = {}

# 记录被解释变量缺失情况
missing_log['NPro缺失数'] = df['NPro'].isnull().sum()
missing_log['STSelf缺失数'] = df['STSelf'].isnull().sum()
missing_log['两者均缺失'] = ((df['NPro'].isna()) & (df['STSelf'].isna())).sum()
missing_log['仅NPro缺失'] = ((df['NPro'].isna()) & (df['STSelf'].notna())).sum()
missing_log['仅STSelf缺失'] = ((df['NPro'].notna()) & (df['STSelf'].isna())).sum()

# 删除被预测变量均缺失的观测
before_drop_y = len(df)
df = df[~((df['NPro'].isna()) & (df['STSelf'].isna()))]
missing_log['删除两者均缺失'] = before_drop_y - len(df)

# 高管核心特征缺失删除
executive_features = [
    'Female_CEO', 'LnAge_CEO', 'ShareRatio_CEO', 'Duality_CEO', 'Parttime_CEO',
    'ProFun_CEO', 'MgtFun_CEO', 'SkiFun_CEO', 'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO',
    'Female_Chair', 'LnAge_Chair', 'ShareRatio_Chair', 'Duality_Chair', 'Parttime_Chair',
    'ProFun_Chair', 'MgtFun_Chair', 'SkiFun_Chair', 'Oversea_Chair', 'AcademicExp_Chair', 'FinBackExp_Chair'
]
existing_exec_feat = [f for f in executive_features if f in df.columns]
before_drop_exec = len(df)
df = df.dropna(subset=existing_exec_feat)
missing_log['删除高管特征缺失'] = before_drop_exec - len(df)

# 控制变量行业-年度均值填充
control_vars = ['Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
                'BoardSize', 'IndepRatio', 'LnFirmAge']
fill_counts = {}
for col in control_vars:
    if col not in df.columns:
        continue
    missing_before = df[col].isnull().sum()
    group_means = df.groupby(['IndustryCode', 'Year'])[col].transform('mean')
    df[col] = df[col].fillna(group_means)
    year_means = df.groupby('Year')[col].transform('mean')
    df[col] = df[col].fillna(year_means)
    missing_after = df[col].isnull().sum()
    fill_counts[col] = missing_before - missing_after

missing_log['控制变量填充详情'] = fill_counts

# 最终缺失检查
final_missing = df.isnull().sum().sum()
if final_missing > 0:
    print(f"  注意：最终数据中仍有 {final_missing} 个缺失值，主要来自被解释变量（NPro/STSelf）的单侧缺失。")
    print("  如需进行回归分析，建议对每个模型分别删除该被解释变量缺失的观测。")

# ==================== 6. 整理最终列顺序 ====================
final_columns = [
    'Stkcd', 'Year', 'IndustryCode',
    'NPro', 'STSelf',
    'Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
    'BoardSize', 'IndepRatio', 'LnFirmAge',
    'Female_CEO', 'LnAge_CEO', 'ShareRatio_CEO', 'Duality_CEO', 'Parttime_CEO',
    'ProFun_CEO', 'MgtFun_CEO', 'SkiFun_CEO', 'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO',
    'Female_Chair', 'LnAge_Chair', 'ShareRatio_Chair', 'Duality_Chair', 'Parttime_Chair',
    'ProFun_Chair', 'MgtFun_Chair', 'SkiFun_Chair', 'Oversea_Chair', 'AcademicExp_Chair', 'FinBackExp_Chair'
]
existing_final_cols = [c for c in final_columns if c in df.columns]
df_final = df[existing_final_cols].copy()

# ==================== 7. 生成报告 ====================
print("生成处理报告...")
report_lines = []
report_lines.append("最终数据合并与后处理报告")
report_lines.append("=" * 60)
report_lines.append("【各表读取行数】")
report_lines.append(f"控制变量表: {len(df_control)}")
report_lines.append(f"新质生产力表: {len(df_npro)}")
report_lines.append(f"科技自立自强表: {len(df_stself)}")
report_lines.append(f"高管特征表: {len(df_executive)}")
report_lines.append("")
report_lines.append("【合并过程观测数变化】")
for step, count in step_counts.items():
    report_lines.append(f"{step}: {count}")
report_lines.append("")
report_lines.append("【被解释变量缺失情况】")
report_lines.append(f"NPro缺失数: {missing_log['NPro缺失数']}")
report_lines.append(f"STSelf缺失数: {missing_log['STSelf缺失数']}")
report_lines.append(f"两者均缺失: {missing_log['两者均缺失']}")
report_lines.append(f"仅NPro缺失: {missing_log['仅NPro缺失']}")
report_lines.append(f"仅STSelf缺失: {missing_log['仅STSelf缺失']}")
report_lines.append("")
report_lines.append("【缺失值处理删除/填充记录】")
report_lines.append(f"删除两者均缺失: {missing_log['删除两者均缺失']}")
report_lines.append(f"删除高管特征缺失: {missing_log['删除高管特征缺失']}")
report_lines.append("控制变量行业-年度均值填充数量:")
for col, num in missing_log['控制变量填充详情'].items():
    report_lines.append(f"  {col}: {num}")
report_lines.append("")
report_lines.append("【最终样本概况】")
report_lines.append(f"最终观测数: {len(df_final)}")
report_lines.append(f"涉及公司数: {df_final['Stkcd'].nunique()}")
report_lines.append(f"年份范围: {df_final['Year'].min()} - {df_final['Year'].max()}")
report_lines.append("")
report_lines.append("【最终变量缺失值数量（NPro/STSelf 可能仍有缺失）】")
for col in df_final.columns:
    miss = df_final[col].isnull().sum()
    if miss > 0:
        report_lines.append(f"{col}: {miss}")
report_lines.append("")
report_lines.append("注：缩尾详细对比请见 winsorization_summary.xlsx")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "final_processing_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

missing_detail_path = os.path.join(OUTPUT_DIR_REPORT, "missing_handling_summary.txt")
with open(missing_detail_path, 'w', encoding='utf-8-sig') as f:
    f.write("缺失值处理详情\n")
    f.write("================\n")
    f.write(f"NPro缺失数: {missing_log['NPro缺失数']}\n")
    f.write(f"STSelf缺失数: {missing_log['STSelf缺失数']}\n")
    f.write(f"两者均缺失: {missing_log['两者均缺失']}\n")
    f.write(f"仅NPro缺失: {missing_log['仅NPro缺失']}\n")
    f.write(f"仅STSelf缺失: {missing_log['仅STSelf缺失']}\n")
    f.write(f"删除两者均缺失: {missing_log['删除两者均缺失']}\n")
    f.write(f"删除高管特征缺失: {missing_log['删除高管特征缺失']}\n")
    f.write("\n控制变量行业-年度均值填充:\n")
    for col, num in missing_log['控制变量填充详情'].items():
        f.write(f"  {col}: {num}\n")

# ==================== 8. 保存最终数据 ====================
print("保存最终数据...")
final_csv_path = os.path.join(OUTPUT_DIR_FINAL, "final_data.csv")
df_final.to_csv(final_csv_path, index=False, encoding='utf-8-sig')

# 尝试保存 feather 格式，若失败则跳过
try:
    final_feather_path = os.path.join(OUTPUT_DIR_FINAL, "final_data.feather")
    df_final.reset_index(drop=True).to_feather(final_feather_path)
    print(f"Feather 格式已保存: {final_feather_path}")
except Exception as e:
    print(f"保存 Feather 失败（可能缺少 pyarrow），已跳过: {e}")

# ==================== 9. 抽样保存 ====================
sample_final = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "final_data_sample.xlsx")
sample_final.to_excel(sample_path, index=False)

# ==================== 10. 控制台摘要 ====================
print("\n最终数据处理完成！")
print(f"最终数据 (CSV) 保存至: {final_csv_path}")
print(f"报告保存至: {report_path}")
print(f"缩尾对比表保存至: {winsor_path}")
print(f"抽样保存至: {sample_path}")
print("\n--- 最终数据摘要 ---")
print(f"观测数: {len(df_final)}")
print(f"公司数: {df_final['Stkcd'].nunique()}")
print(f"年份范围: {df_final['Year'].min()} - {df_final['Year'].max()}")
print(f"NPro 缺失数: {df_final['NPro'].isnull().sum()}")
print(f"STSelf 缺失数: {df_final['STSelf'].isnull().sum()}")