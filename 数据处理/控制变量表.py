import pandas as pd
import numpy as np
import os

# ==================== 配置参数 ====================
INPUT_DIR = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_TABLES = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表的报告与抽样"

# 输入文件路径
FIN_FILE = os.path.join(INPUT_DIR, "FS_Combas_cleaned.csv")
SHARES_FILE = os.path.join(INPUT_DIR, "股本结构_cleaned.csv")
TOP10_FILE = os.path.join(INPUT_DIR, "大股东持股_cleaned.csv")
BOARD_FILE = os.path.join(INPUT_DIR, "TMT_POSITION_board_cleaned.csv")
INFO_FILE = os.path.join(INPUT_DIR, "CSp_ListedCoInfoAnl_cleaned.csv")

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取输入文件...")
df_fin = pd.read_csv(FIN_FILE, dtype={'Stkcd': str})
df_shares = pd.read_csv(SHARES_FILE, dtype={'Stkcd': str})
df_top10 = pd.read_csv(TOP10_FILE, dtype={'Stkcd': str})
df_board = pd.read_csv(BOARD_FILE, dtype={'Stkcd': str})
df_info = pd.read_csv(INFO_FILE, dtype={'Stkcd': str})

print(f"财务表行数: {len(df_fin)}")
print(f"股本结构表行数: {len(df_shares)}")
print(f"大股东持股表行数: {len(df_top10)}")
print(f"董事会特征表行数: {len(df_board)}")
print(f"公司基本信息表行数: {len(df_info)}")

# 统一年份类型
for dataframe in [df_fin, df_shares, df_top10, df_board]:
    if 'Year' in dataframe.columns:
        dataframe['Year'] = dataframe['Year'].astype(int)

# ==================== 步骤 2.6.1 - 2.6.5：左连接各表 ====================
print("\n开始合并各表...")
# 以财务表为基准
df = df_fin.copy()
step_records = {'基准财务表': len(df)}

# 连接股本结构表
df = df.merge(df_shares, on=['Stkcd', 'Year'], how='left')
step_records['连接股本结构后'] = len(df)

# 连接大股东持股表
df = df.merge(df_top10, on=['Stkcd', 'Year'], how='left')
step_records['连接大股东持股后'] = len(df)

# 连接董事会特征表
df = df.merge(df_board, on=['Stkcd', 'Year'], how='left')
step_records['连接董事会特征后'] = len(df)

# 连接公司基本信息表（截面数据，仅用 Stkcd 连接）
if df_info['Stkcd'].duplicated().any():
    print("警告：公司基本信息表存在重复 Stkcd，将保留第一条记录。")
    df_info = df_info.drop_duplicates(subset=['Stkcd'], keep='first')
df = df.merge(df_info, on='Stkcd', how='left')
step_records['连接公司基本信息后'] = len(df)

# ==================== 步骤 2.6.6 - 2.6.7：计算公司年龄 ====================
df['EstYear'] = pd.to_numeric(df['EstYear'], errors='coerce')
df['FirmAge'] = df['Year'] - df['EstYear'] + 1
df.loc[df['FirmAge'] <= 0, 'FirmAge'] = np.nan
df['LnFirmAge'] = np.log(df['FirmAge'])

# 记录年龄计算后的缺失情况
firmage_valid = df['FirmAge'].notna().sum()
lnfirmage_missing = df['LnFirmAge'].isnull().sum()

# ==================== 步骤 2.6.8：样本筛选 ====================
print("\n开始样本筛选...")
filter_counts = {}

# ① 删除 ST 公司
before_st = len(df)
df = df[df['IsST'] != 1]
filter_counts['删除 ST 公司'] = before_st - len(df)

# ② 删除金融/房地产行业
before_finreal = len(df)
df = df[df['IsFinReal'] != 1]
filter_counts['删除金融/房地产'] = before_finreal - len(df)

# ③ 删除上市前数据（Year < ListYear）
df['ListYear'] = pd.to_numeric(df['ListYear'], errors='coerce')
before_list = len(df)
df = df[~((df['Year'] < df['ListYear']) & df['ListYear'].notna())]
filter_counts['删除上市前数据'] = before_list - len(df)

# ==================== 步骤 2.6.9：最终保留列 ====================
final_columns = [
    'Stkcd', 'Year', 'Size', 'Leverage', 'PPE', 'StateShare',
    'Top10HoldRatio', 'BoardSize', 'IndepRatio', 'LnFirmAge',
    'IndustryCode', 'TotalShares'
]
for col in final_columns:
    if col not in df.columns:
        print(f"警告：最终列 '{col}' 不在合并表中，将填充为 NaN")
        df[col] = np.nan

df_final = df[final_columns].copy()
final_rows = len(df_final)

# ==================== 生成报告 ====================
print("\n正在生成报告...")
report_lines = []
report_lines.append("控制变量表构造报告")
report_lines.append("=" * 60)
report_lines.append("【输入表行数】")
report_lines.append(f"财务表: {len(df_fin)}")
report_lines.append(f"股本结构表: {len(df_shares)}")
report_lines.append(f"大股东持股表: {len(df_top10)}")
report_lines.append(f"董事会特征表: {len(df_board)}")
report_lines.append(f"公司基本信息表: {len(df_info)}")
report_lines.append("")

report_lines.append("【连接步骤观测数】")
for step, count in step_records.items():
    report_lines.append(f"{step}: {count}")
report_lines.append("")

report_lines.append("【公司年龄计算】")
report_lines.append(f"FirmAge 有效观测数: {firmage_valid}")
report_lines.append(f"LnFirmAge 缺失数: {lnfirmage_missing}")
report_lines.append("")

report_lines.append("【样本筛选删除记录数】")
for reason, count in filter_counts.items():
    report_lines.append(f"{reason}: {count}")
report_lines.append("")

report_lines.append("【最终数据集概况】")
report_lines.append(f"最终观测数: {final_rows}")
report_lines.append(f"涉及公司数: {df_final['Stkcd'].nunique()}")
report_lines.append(f"年份范围: {df_final['Year'].min()} - {df_final['Year'].max()}")
report_lines.append("")

report_lines.append("【连续变量描述统计】")
numeric_cols = ['Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
                'BoardSize', 'IndepRatio', 'LnFirmAge']
for col in numeric_cols:
    if col in df_final.columns:
        report_lines.append(f"\n{col}:")
        report_lines.append(df_final[col].describe().to_string())
    else:
        report_lines.append(f"\n{col}: 列不存在")

report_lines.append("\n【各变量缺失值数量】")
missing_report = df_final.isnull().sum()
for col in final_columns:
    report_lines.append(f"{col}: {missing_report[col]}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "control_construction_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存数据表 ====================
control_path = os.path.join(OUTPUT_DIR_TABLES, "df_control.csv")
df_final.to_csv(control_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample_control = df_final.sample(n=min(SAMPLE_SIZE, len(df_final)), random_state=SEED)
sample_path = os.path.join(OUTPUT_DIR_REPORT, "df_control_sample.xlsx")
sample_control.to_excel(sample_path, index=False)

# ==================== 控制台摘要 ====================
print("\n控制变量表构造完成！")
print(f"控制变量表保存至: {control_path}")
print(f"报告保存至: {report_path}")
print(f"抽样保存至: {sample_path}")
print("\n--- 快速摘要 ---")
print(f"最终观测数: {final_rows}")
print(f"涉及公司数: {df_final['Stkcd'].nunique()}")
print(f"年份范围: {df_final['Year'].min()} - {df_final['Year'].max()}")
print(f"删除 ST 公司: {filter_counts['删除 ST 公司']}")
print(f"删除金融/房地产: {filter_counts['删除金融/房地产']}")
print(f"删除上市前数据: {filter_counts['删除上市前数据']}")