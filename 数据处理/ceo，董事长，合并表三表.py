import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 警告（本任务不涉及 Excel 读取，但为保持统一）
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_DIR = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_TABLES = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第二次处理-构造4个表\构造的表的报告与抽样"

CEO_ID_FILE = os.path.join(INPUT_DIR, "TMT_POSITION_ceo_id_cleaned.csv")
CHAIR_ID_FILE = os.path.join(INPUT_DIR, "TMT_POSITION_chair_id_cleaned.csv")
FEATURE_FILE = os.path.join(INPUT_DIR, "TMT_FIGUREINFO_cleaned.csv")

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_TABLES, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取数据 ====================
print("正在读取输入文件...")
df_ceo_id = pd.read_csv(CEO_ID_FILE, dtype={'Stkcd': str, 'PersonID_CEO': str})
df_chair_id = pd.read_csv(CHAIR_ID_FILE, dtype={'Stkcd': str, 'PersonID_Chair': str})
df_feature = pd.read_csv(FEATURE_FILE, dtype={'Stkcd': str, 'PersonID': str})

print(f"CEO ID 表行数: {len(df_ceo_id)}")
print(f"董事长 ID 表行数: {len(df_chair_id)}")
print(f"高管特征池行数: {len(df_feature)}")

# 统一年份数据类型
df_ceo_id['Year'] = df_ceo_id['Year'].astype(int)
df_chair_id['Year'] = df_chair_id['Year'].astype(int)
df_feature['Year'] = df_feature['Year'].astype(int)

# ==================== 步骤 5.3：匹配 CEO 特征 ====================
print("\n正在匹配 CEO 特征...")
# 左连接：CEO ID 表 + 特征池
df_ceo_merged = df_ceo_id.merge(
    df_feature,
    left_on=['Stkcd', 'Year', 'PersonID_CEO'],
    right_on=['Stkcd', 'Year', 'PersonID'],
    how='left',
    suffixes=('', '_feature')  # 避免列名冲突
)

# 选取需要重命名的特征列
ceo_feature_cols = [
    'Female', 'LnAge', 'SharEnd', 'IsDuality', 'Parttime',
    'ProFun', 'MgtFun', 'SkiFun', 'Oversea', 'AcademicExp', 'FinBackExp'
]
# 检查列是否存在
missing_ceo = set(ceo_feature_cols) - set(df_ceo_merged.columns)
if missing_ceo:
    raise KeyError(f"CEO 匹配时缺少特征列: {missing_ceo}")

# 重命名为 _CEO 后缀
rename_ceo = {col: f"{col}_CEO" for col in ceo_feature_cols}
# 特殊处理：IsDuality 重命名为 Duality_CEO
rename_ceo['IsDuality'] = 'Duality_CEO'
df_ceo_merged.rename(columns=rename_ceo, inplace=True)

# 保留主键 + 重命名后的列
ceo_final_cols = ['Stkcd', 'Year'] + list(rename_ceo.values())
df_ceo_final = df_ceo_merged[ceo_final_cols].copy()

# ==================== 步骤 5.4：匹配董事长特征 ====================
print("正在匹配董事长特征...")
df_chair_merged = df_chair_id.merge(
    df_feature,
    left_on=['Stkcd', 'Year', 'PersonID_Chair'],
    right_on=['Stkcd', 'Year', 'PersonID'],
    how='left',
    suffixes=('', '_feature')
)

# 选取并重命名特征列
chair_feature_cols = ceo_feature_cols  # 相同字段
rename_chair = {col: f"{col}_Chair" for col in chair_feature_cols}
rename_chair['IsDuality'] = 'Duality_Chair'
df_chair_merged.rename(columns=rename_chair, inplace=True)

chair_final_cols = ['Stkcd', 'Year'] + list(rename_chair.values())
df_chair_final = df_chair_merged[chair_final_cols].copy()

# ==================== 步骤 5.5：合并 CEO 与董事长特征 ====================
print("正在合并 CEO 与董事长特征...")
df_executive = df_ceo_final.merge(
    df_chair_final,
    on=['Stkcd', 'Year'],
    how='left'
)

# ==================== 生成报告 ====================
def missing_summary(df, label, cols):
    """生成缺失统计字符串"""
    lines = []
    total = len(df)
    for col in cols:
        miss = df[col].isnull().sum()
        pct = miss / total if total > 0 else 0
        lines.append(f"  {col}: 缺失 {miss} ({pct:.2%})")
    return '\n'.join(lines)

# 两职合一一致性核查
duality_check = df_executive[['Duality_CEO', 'Duality_Chair']].copy()
both_notna = duality_check.dropna()
inconsistent = (both_notna['Duality_CEO'] != both_notna['Duality_Chair']).sum()
both_na = duality_check.isnull().all(axis=1).sum()
only_ceo = duality_check['Duality_CEO'].notna() & duality_check['Duality_Chair'].isna()
only_chair = duality_check['Duality_CEO'].isna() & duality_check['Duality_Chair'].notna()

report_lines = []
report_lines.append("高管特征匹配报告")
report_lines.append("=" * 60)
report_lines.append(f"CEO ID 表行数: {len(df_ceo_id)}")
report_lines.append(f"董事长 ID 表行数: {len(df_chair_id)}")
report_lines.append(f"高管特征池行数: {len(df_feature)}")
report_lines.append("")

report_lines.append("--- CEO 特征表 ---")
report_lines.append(f"最终观测数: {len(df_ceo_final)}")
report_lines.append(f"涉及公司数: {df_ceo_final['Stkcd'].nunique()}")
report_lines.append(f"年份范围: {df_ceo_final['Year'].min()} - {df_ceo_final['Year'].max()}")
report_lines.append("各特征缺失情况:")
report_lines.append(missing_summary(df_ceo_final, "CEO", rename_ceo.values()))
report_lines.append("")

report_lines.append("--- 董事长特征表 ---")
report_lines.append(f"最终观测数: {len(df_chair_final)}")
report_lines.append(f"涉及公司数: {df_chair_final['Stkcd'].nunique()}")
report_lines.append(f"年份范围: {df_chair_final['Year'].min()} - {df_chair_final['Year'].max()}")
report_lines.append("各特征缺失情况:")
report_lines.append(missing_summary(df_chair_final, "Chair", rename_chair.values()))
report_lines.append("")

report_lines.append("--- 合并表 ---")
report_lines.append(f"最终观测数: {len(df_executive)}")
report_lines.append("各特征缺失情况 (CEO 侧):")
ceo_side_cols = list(rename_ceo.values())
report_lines.append(missing_summary(df_executive, "CEO", ceo_side_cols))
report_lines.append("各特征缺失情况 (Chair 侧):")
chair_side_cols = list(rename_chair.values())
report_lines.append(missing_summary(df_executive, "Chair", chair_side_cols))
report_lines.append("")

report_lines.append("--- 两职合一一致性核查 ---")
report_lines.append(f"CEO 与董事长特征均非空的观测数: {len(both_notna)}")
report_lines.append(f"其中不一致的观测数: {inconsistent}")
report_lines.append(f"仅 CEO 有值的观测数: {only_ceo.sum()}")
report_lines.append(f"仅董事长有值的观测数: {only_chair.sum()}")
report_lines.append(f"两者均缺失的观测数: {both_na}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "executive_construction_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存数据表 ====================
print("\n正在保存输出文件...")
ceo_path = os.path.join(OUTPUT_DIR_TABLES, "df_ceo_final.csv")
df_ceo_final.to_csv(ceo_path, index=False, encoding='utf-8-sig')

chair_path = os.path.join(OUTPUT_DIR_TABLES, "df_chairman_final.csv")
df_chair_final.to_csv(chair_path, index=False, encoding='utf-8-sig')

executive_path = os.path.join(OUTPUT_DIR_TABLES, "df_executive.csv")
df_executive.to_csv(executive_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample_ceo = df_ceo_final.sample(n=min(SAMPLE_SIZE, len(df_ceo_final)), random_state=SEED)
sample_ceo_path = os.path.join(OUTPUT_DIR_REPORT, "df_ceo_final_sample.xlsx")
sample_ceo.to_excel(sample_ceo_path, index=False)

sample_chair = df_chair_final.sample(n=min(SAMPLE_SIZE, len(df_chair_final)), random_state=SEED)
sample_chair_path = os.path.join(OUTPUT_DIR_REPORT, "df_chairman_final_sample.xlsx")
sample_chair.to_excel(sample_chair_path, index=False)

sample_exec = df_executive.sample(n=min(SAMPLE_SIZE, len(df_executive)), random_state=SEED)
sample_exec_path = os.path.join(OUTPUT_DIR_REPORT, "df_executive_sample.xlsx")
sample_exec.to_excel(sample_exec_path, index=False)

# ==================== 控制台摘要 ====================
print("\n匹配完成！")
print(f"CEO 特征表保存至: {ceo_path}")
print(f"董事长特征表保存至: {chair_path}")
print(f"合并表保存至: {executive_path}")
print(f"报告保存至: {report_path}")
print("\n--- 快速摘要 ---")
print(f"CEO 表观测数: {len(df_ceo_final)}")
print(f"董事长表观测数: {len(df_chair_final)}")
print(f"合并表观测数: {len(df_executive)}")