import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置路径 ====================
INPUT_EXCEL = r"D:\how dare you\2026统计建模\数据\公司基本信息\公司基本信息.xlsx"
INPUT_CONTROL = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表\df_control.csv"
INPUT_FINAL = r"D:\how dare you\2026统计建模\数据处理\第三次处理-最终数据\final_data.csv"

OUTPUT_DIR_SUPP = r"D:\how dare you\2026统计建模\数据处理\第三次处理-补充公司信息\清洗后的补充表"
OUTPUT_DIR_CONTROL = r"D:\how dare you\2026统计建模\数据处理\第三次处理-补充公司信息\更新后的控制变量表"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第三次处理-补充公司信息\报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

for d in [OUTPUT_DIR_SUPP, OUTPUT_DIR_CONTROL, OUTPUT_DIR_REPORT]:
    os.makedirs(d, exist_ok=True)

# ==================== 阶段一：清洗补充公司基本信息表 ====================
print("=" * 50)
print("阶段一：清洗补充公司基本信息表")
print("=" * 50)

df_raw = pd.read_excel(INPUT_EXCEL, usecols=['Scode', 'Estbdt', 'Listdt', 'IndcodeB'], dtype={'Scode': str})
print(f"原始行数: {len(df_raw)}")

df_info = df_raw.copy()
df_info.rename(columns={'Scode': 'Stkcd', 'Estbdt': 'EstablishDate', 'Listdt': 'ListingDate', 'IndcodeB': 'IndustryCode'}, inplace=True)

# 股票代码补零
df_info['Stkcd'] = df_info['Stkcd'].str.zfill(6)

# 提取年份
df_info['EstYear'] = pd.to_datetime(df_info['EstablishDate'], errors='coerce').dt.year
df_info['ListYear'] = pd.to_datetime(df_info['ListingDate'], errors='coerce').dt.year

# 保留必要列
df_info = df_info[['Stkcd', 'EstYear', 'ListYear', 'IndustryCode']]

# 按 Stkcd 去重：优先保留 EstYear 和 ListYear 非缺失的记录，再取第一条
df_info = df_info.sort_values(['Stkcd', 'EstYear', 'ListYear'], na_position='last')
df_info = df_info.drop_duplicates(subset=['Stkcd'], keep='first')

print(f"去重后公司数: {len(df_info)}")

# 缺失统计
missing_stats = df_info[['EstYear', 'ListYear', 'IndustryCode']].isnull().mean() * 100

# 保存清洗后补充表
supp_cleaned_path = os.path.join(OUTPUT_DIR_SUPP, "company_info_supplement_cleaned.csv")
df_info.to_csv(supp_cleaned_path, index=False, encoding='utf-8-sig')
print(f"补充表清洗完成，保存至: {supp_cleaned_path}")

# 抽样
sample_info = df_info.sample(n=min(SAMPLE_SIZE, len(df_info)), random_state=SEED)
sample_info_path = os.path.join(OUTPUT_DIR_REPORT, "company_info_supplement_sample.xlsx")
sample_info.to_excel(sample_info_path, index=False)

# ==================== 阶段二：更新控制变量表 ====================
print("\n" + "=" * 50)
print("阶段二：更新控制变量表")
print("=" * 50)

df_control = pd.read_csv(INPUT_CONTROL, dtype={'Stkcd': str, 'IndustryCode': str})
print(f"原控制变量表行数: {len(df_control)}")
old_missing_lnage = df_control['LnFirmAge'].isnull().sum()
old_missing_ind = df_control['IndustryCode'].isnull().sum()
print(f"原 LnFirmAge 缺失数: {old_missing_lnage}")
print(f"原 IndustryCode 缺失数: {old_missing_ind}")

# 删除旧信息列（IndustryCode 和 LnFirmAge）
df_control = df_control.drop(columns=['IndustryCode', 'LnFirmAge'], errors='ignore')

# 左连接补充信息
df_control = df_control.merge(df_info, on='Stkcd', how='left')

# 重新计算公司年龄
df_control['FirmAge'] = df_control['Year'] - df_control['EstYear'] + 1
df_control.loc[df_control['FirmAge'] <= 0, 'FirmAge'] = np.nan
df_control['LnFirmAge'] = np.log(df_control['FirmAge'])

# 删除上市前数据（Year < ListYear）
before_list = len(df_control)
df_control['ListYear'] = pd.to_numeric(df_control['ListYear'], errors='coerce')
df_control = df_control[~((df_control['Year'] < df_control['ListYear']) & df_control['ListYear'].notna())]
after_list = len(df_control)
dropped_list = before_list - after_list

# 最终保留列（原控制变量表列顺序）
final_cols = ['Stkcd', 'Year', 'Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
              'BoardSize', 'IndepRatio', 'LnFirmAge', 'IndustryCode', 'TotalShares']
df_control_updated = df_control[final_cols].copy()

new_missing_lnage = df_control_updated['LnFirmAge'].isnull().sum()
new_missing_ind = df_control_updated['IndustryCode'].isnull().sum()
matched_companies = df_control_updated['Stkcd'].isin(df_info['Stkcd']).sum()
fill_rate_est = 1 - (df_control_updated['LnFirmAge'].isnull().sum() / len(df_control_updated))

print(f"连接补充表后，匹配到信息的公司数: {df_info['Stkcd'].nunique()} 家中的 {df_control_updated['Stkcd'].nunique()} 家")
print(f"更新后 LnFirmAge 缺失数: {new_missing_lnage} (填补比例: {fill_rate_est:.2%})")
print(f"更新后 IndustryCode 缺失数: {new_missing_ind}")
print(f"删除上市前数据: {dropped_list} 行")

# 保存更新后的控制变量表
control_updated_path = os.path.join(OUTPUT_DIR_CONTROL, "df_control_updated.csv")
df_control_updated.to_csv(control_updated_path, index=False, encoding='utf-8-sig')
print(f"更新后控制变量表保存至: {control_updated_path}")

# 抽样
sample_control = df_control_updated.sample(n=min(SAMPLE_SIZE, len(df_control_updated)), random_state=SEED)
sample_control_path = os.path.join(OUTPUT_DIR_REPORT, "df_control_updated_sample.xlsx")
sample_control.to_excel(sample_control_path, index=False)

# ==================== 阶段三：重新生成最终分析数据 ====================
print("\n" + "=" * 50)
print("阶段三：更新最终分析数据")
print("=" * 50)

df_final = pd.read_csv(INPUT_FINAL, dtype={'Stkcd': str, 'IndustryCode': str})
print(f"原最终数据行数: {len(df_final)}")

# 删除旧的行业代码和年龄对数
df_final = df_final.drop(columns=['IndustryCode', 'LnFirmAge'], errors='ignore')

# 左连接更新后的信息（仅取 Stkcd, Year, LnFirmAge, IndustryCode）
df_update_info = df_control_updated[['Stkcd', 'Year', 'LnFirmAge', 'IndustryCode']]
df_final_updated = df_final.merge(df_update_info, on=['Stkcd', 'Year'], how='left')

# 按原列顺序重新排列（假定原列顺序中存在 IndustryCode 和 LnFirmAge 位置）
# 简单处理：将 IndustryCode 和 LnFirmAge 放到末尾或原位置均可，此处直接保留 merge 后的列顺序
print(f"更新后最终数据行数: {len(df_final_updated)}")

# 保存
final_csv_path = os.path.join(OUTPUT_DIR_CONTROL, "final_data_updated.csv")
df_final_updated.to_csv(final_csv_path, index=False, encoding='utf-8-sig')
final_feather_path = os.path.join(OUTPUT_DIR_CONTROL, "final_data_updated.feather")
df_final_updated.reset_index(drop=True).to_feather(final_feather_path)

# 抽样
sample_final = df_final_updated.sample(n=min(SAMPLE_SIZE, len(df_final_updated)), random_state=SEED)
sample_final_path = os.path.join(OUTPUT_DIR_REPORT, "final_data_updated_sample.xlsx")
sample_final.to_excel(sample_final_path, index=False)

# ==================== 生成报告 ====================
print("\n生成综合报告...")
report_lines = []
report_lines.append("公司基本信息补充与更新报告")
report_lines.append("=" * 60)
report_lines.append("【补充表清洗概况】")
report_lines.append(f"原始行数: {len(df_raw)}")
report_lines.append(f"去重后公司数: {len(df_info)}")
report_lines.append(f"EstYear 缺失比例: {missing_stats['EstYear']:.2f}%")
report_lines.append(f"ListYear 缺失比例: {missing_stats['ListYear']:.2f}%")
report_lines.append(f"IndustryCode 缺失比例: {missing_stats['IndustryCode']:.2f}%")
report_lines.append("")
report_lines.append("【控制变量表更新前后对比】")
report_lines.append(f"更新前观测数: {len(df_control)}")
report_lines.append(f"更新前 LnFirmAge 缺失: {old_missing_lnage}")
report_lines.append(f"更新前 IndustryCode 缺失: {old_missing_ind}")
report_lines.append(f"更新后观测数: {len(df_control_updated)}")
report_lines.append(f"更新后 LnFirmAge 缺失: {new_missing_lnage}")
report_lines.append(f"更新后 IndustryCode 缺失: {new_missing_ind}")
report_lines.append(f"删除上市前数据行数: {dropped_list}")
report_lines.append(f"LnFirmAge 填补率: {fill_rate_est:.2%}")
report_lines.append("")
report_lines.append("【最终数据更新】")
report_lines.append(f"原最终数据行数: {len(df_final)}")
report_lines.append(f"更新后最终数据行数: {len(df_final_updated)}")
report_lines.append("")
report_lines.append("注：所有缺失值均已在缩尾与填充阶段处理完毕，本次更新后仍可能存在少量缺失，建议在后续分析中酌情处理。")

report_path = os.path.join(OUTPUT_DIR_REPORT, "supplement_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

print("\n全部更新完成！")
print(f"报告保存至: {report_path}")
print(f"更新后控制变量表: {control_updated_path}")
print(f"更新后最终数据: {final_csv_path}")
print(f"抽样文件位于: {OUTPUT_DIR_REPORT}")