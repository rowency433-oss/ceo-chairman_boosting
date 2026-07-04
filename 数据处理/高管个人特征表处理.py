import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE_1 = r"D:\how dare you\2026统计建模\数据\董监高个人特征文件162730779(仅供北洋大学使用)\TMT_FIGUREINFO.xlsx"
INPUT_FILE_2 = r"D:\how dare you\2026统计建模\数据\董监高个人特征文件162730779(仅供北洋大学使用)\TMT_FIGUREINFO1.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取并合并两个文件 ====================
print("正在读取 TMT_FIGUREINFO.xlsx...")
df1 = pd.read_excel(INPUT_FILE_1, dtype={'Stkcd': str, 'PersonID': str})
rows1 = len(df1)
print(f"文件1行数: {rows1}")

print("正在读取 TMT_FIGUREINFO1.xlsx...")
df2 = pd.read_excel(INPUT_FILE_2, dtype={'Stkcd': str, 'PersonID': str})
rows2 = len(df2)
print(f"文件2行数: {rows2}")

# 纵向拼接
df_raw = pd.concat([df1, df2], ignore_index=True)
raw_rows = len(df_raw)
print(f"合并后总行数: {raw_rows}")

# ==================== 第一阶段：数据准备与筛选 ====================

# 步骤2：提取年份
df_raw['Year'] = pd.to_datetime(df_raw['Reptdt'], errors='coerce').dt.year
df_raw.dropna(subset=['Year'], inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# 步骤3：筛选年份范围
df = df_raw[(df_raw['Year'] >= 2008) & (df_raw['Year'] <= 2024)].copy()
after_year_filter = len(df)

# 步骤4：格式化股票代码
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 步骤5：同一人同一年取最新记录（按 Reptdt 降序）
# 为确保 Reptdt 可用于排序，先转为 datetime
df['Reptdt_dt'] = pd.to_datetime(df['Reptdt'], errors='coerce')
df_sorted = df.sort_values(['Stkcd', 'Year', 'PersonID', 'Reptdt_dt'],
                           ascending=[True, True, True, False])
df = df_sorted.drop_duplicates(subset=['Stkcd', 'Year', 'PersonID'], keep='first').copy()
after_dedup = len(df)

# 步骤6：保留必要字段
required_cols = ['Stkcd', 'Year', 'PersonID', 'Gender', 'Age', 'SharEnd',
                 'IsDuality', 'Funback', 'OveseaBack', 'Academic', 'FinBack',
                 'IsCocurP', 'Director_ListCO']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# ==================== 第二阶段：多值字段解析 ====================

def parse_multi_value(series, target_values):
    """
    将逗号分隔的字符串字段解析为是否包含目标值的布尔序列。
    series: pandas Series，可能包含 NaN、空字符串、数值或逗号分隔的字符串。
    target_values: 列表或集合，要检查的目标值（字符串形式）。
    """
    # 转为字符串，NaN 填充为空字符串
    s = series.fillna('').astype(str)
    # 拆分为列表，并检查与目标值的交集是否非空
    def check_contains(val):
        if val == '':
            return False
        parts = set(val.split(','))
        return bool(parts.intersection(target_values))
    return s.apply(check_contains).astype(int)

# 步骤7-9：Funback 解析
# 生产/研发/设计背景：包含 '1','2','3'
df['ProFun'] = parse_multi_value(df['Funback'], {'1', '2', '3'})
# 管理/市场/人力背景：包含 '4','5','6'
df['MgtFun'] = parse_multi_value(df['Funback'], {'4', '5', '6'})
# 财务/法律背景：包含 '8','9'
df['SkiFun'] = parse_multi_value(df['Funback'], {'8', '9'})

# 步骤10：OveseaBack 解析
# 海外背景：不为空且不包含 '3'（无海外背景）
def has_oversea(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str == '':
        return 0
    parts = set(val_str.split(','))
    if '3' in parts:
        return 0
    return 1
df['Oversea'] = df['OveseaBack'].apply(has_oversea)

# 步骤11：AcademicExp 解析
# 学术背景：不为空且不为 '4'
def has_academic(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str == '' or val_str == '4':
        return 0
    return 1
df['AcademicExp'] = df['Academic'].apply(has_academic)

# 步骤12：FinBackExp 解析
# 金融背景：不为空且不为 '99'
def has_finback(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str == '' or val_str == '99':
        return 0
    return 1
df['FinBackExp'] = df['FinBack'].apply(has_finback)

# 步骤13：Parttime 解析
# IsCocurP == 1 或 Director_ListCO 非空且非 '0'
def has_parttime(row):
    if pd.notna(row['IsCocurP']) and row['IsCocurP'] == 1:
        return 1
    dir_val = row['Director_ListCO']
    if pd.notna(dir_val):
        val_str = str(dir_val).strip()
        if val_str != '' and val_str != '0':
            return 1
    return 0
df['Parttime'] = df.apply(has_parttime, axis=1)

# ==================== 第三阶段：连续变量处理 ====================

# 步骤14：年龄对数
# 将 Age 转为数值，无效值转为 NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['LnAge'] = np.log(df['Age'].where(df['Age'] > 0))

# 步骤15：女性标识
# Gender 字段可能为 "男"/"女"，也可能为英文，统一判断
df['Female'] = df['Gender'].astype(str).str.contains('女|F|f', na=False).astype(int)

# ==================== 第四阶段：最终保留列 ====================
final_cols = [
    'Stkcd', 'Year', 'PersonID', 'Female', 'LnAge', 'SharEnd',
    'IsDuality', 'Parttime', 'ProFun', 'MgtFun', 'SkiFun',
    'Oversea', 'AcademicExp', 'FinBackExp'
]
df_final = df[final_cols].copy()

# ==================== 统计信息 ====================
final_rows = len(df_final)
unique_stocks = df_final['Stkcd'].nunique()
unique_persons = df_final['PersonID'].nunique()
year_range = f"{df_final['Year'].min()} - {df_final['Year'].max()}"

# 各衍生变量的取值分布
binary_vars = ['Female', 'ProFun', 'MgtFun', 'SkiFun', 'Oversea',
               'AcademicExp', 'FinBackExp', 'Parttime']
distributions = {}
for var in binary_vars:
    counts = df_final[var].value_counts()
    total = len(df_final)
    dist_0 = counts.get(0, 0)
    dist_1 = counts.get(1, 0)
    distributions[var] = (dist_0, dist_1, dist_0/total, dist_1/total)

# 缺失值统计
missing_counts = df_final.isnull().sum()

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("TMT_FIGUREINFO 表清洗报告")
report_lines.append("=" * 60)
report_lines.append(f"文件1原始行数: {rows1}")
report_lines.append(f"文件2原始行数: {rows2}")
report_lines.append(f"合并后总行数: {raw_rows}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"去重（同人同年取最新）后行数: {after_dedup}")
report_lines.append(f"最终观测数（高管-年度）: {final_rows}")
report_lines.append(f"涉及公司数: {unique_stocks}")
report_lines.append(f"涉及高管人数: {unique_persons}")
report_lines.append(f"年份范围: {year_range}")

report_lines.append("\n--- 衍生变量分布 ---")
for var in binary_vars:
    d0, d1, p0, p1 = distributions[var]
    report_lines.append(f"{var}: 0={d0} ({p0:.2%}), 1={d1} ({p1:.2%})")

report_lines.append("\n--- 连续变量描述统计 ---")
report_lines.append("LnAge 描述统计:")
report_lines.append(df_final['LnAge'].describe().to_string())
report_lines.append("\nSharEnd 描述统计:")
# 将 SharEnd 转为数值以便描述
shar_num = pd.to_numeric(df_final['SharEnd'], errors='coerce')
report_lines.append(shar_num.describe().to_string())

report_lines.append("\n--- 各变量缺失值数量 ---")
for col in final_cols:
    report_lines.append(f"{col}: {missing_counts[col]}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "TMT_FIGUREINFO_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
cleaned_path = os.path.join(OUTPUT_DIR_CLEANED, "TMT_FIGUREINFO_cleaned.csv")
df_final.to_csv(cleaned_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "TMT_FIGUREINFO_sample.xlsx")
sample.to_excel(sample_path, index=False)

# ==================== 控制台输出摘要 ====================
print("\n清洗完成！")
print(f"清洗后数据已保存至: {cleaned_path}")
print(f"报告已保存至: {report_path}")
print(f"抽样数据已保存至: {sample_path}")
print("\n--- 报告摘要 ---")
print(f"最终观测数: {final_rows}")
print(f"涉及公司数: {unique_stocks}")
print(f"涉及高管人数: {unique_persons}")
print(f"年份范围: {year_range}")