import pandas as pd
import numpy as np
import os
import warnings

# 屏蔽 openpyxl 默认样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# ==================== 配置参数 ====================
INPUT_FILE_1 = r"D:\how dare you\2026统计建模\数据\董监高任职情况表103859776(仅供UC Berkeley使用)\TMT_POSITION.xlsx"
INPUT_FILE_2 = r"D:\how dare you\2026统计建模\数据\董监高任职情况表103859776(仅供UC Berkeley使用)\TMT_POSITION1.xlsx"
OUTPUT_DIR_CLEANED = r"D:\how dare you\2026统计建模\数据处理\第一次处理"
OUTPUT_DIR_REPORT = r"D:\how dare you\2026统计建模\数据处理\第一次处理报告与抽样"

SEED = 42
SAMPLE_SIZE = 10

# 创建输出目录
os.makedirs(OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(OUTPUT_DIR_REPORT, exist_ok=True)

# ==================== 读取并合并两个文件 ====================
print("正在读取 TMT_POSITION.xlsx...")
df1 = pd.read_excel(INPUT_FILE_1, dtype={'Stkcd': str, 'PersonID': str})
rows1 = len(df1)
print(f"文件1行数: {rows1}")

print("正在读取 TMT_POSITION1.xlsx...")
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

# 步骤4：筛选在任状态（若存在ServiceStatus字段且非空，则要求等于1；否则忽略）
if 'ServiceStatus' in df.columns:
    # 将 ServiceStatus 转为数值，无法转换的置为 NaN
    df['ServiceStatus'] = pd.to_numeric(df['ServiceStatus'], errors='coerce')
    # 仅保留在职（值为1）或缺失的记录（缺失视为有效）
    df = df[(df['ServiceStatus'] == 1) | (df['ServiceStatus'].isna())]
after_status_filter = len(df)

# 步骤5：格式化股票代码
df['Stkcd'] = df['Stkcd'].str.zfill(6)

# 步骤6：保留必要字段
required_cols = ['Stkcd', 'Year', 'PersonID', 'GTAPosition', 'Position']
missing_cols = set(required_cols) - set(df.columns)
if missing_cols:
    raise KeyError(f"缺少必要字段: {missing_cols}")
df = df[required_cols]

# 为便于后续筛选，将职务字段统一转为小写副本
df['GTAPosition_lower'] = df['GTAPosition'].fillna('').astype(str).str.lower()
df['Position_lower'] = df['Position'].fillna('').astype(str).str.lower()

# ==================== 第二阶段：董事会特征构造 ====================

# 步骤7：计算董事会人数 BoardSize
# 筛选所有包含"董事"的职务（含董事长、副董事长、董事、独立董事等）
is_director = df['GTAPosition_lower'].str.contains('董事', na=False)
director_df = df[is_director].copy()

# 按 Stkcd, Year 分组，统计唯一 PersonID 数量
board_size = director_df.groupby(['Stkcd', 'Year'])['PersonID'].nunique().reset_index()
board_size.rename(columns={'PersonID': 'BoardSize'}, inplace=True)

# 步骤8：计算独立董事人数 IndepNum
is_indep = director_df['GTAPosition_lower'] == '独立董事'
indep_df = director_df[is_indep].copy()
indep_num = indep_df.groupby(['Stkcd', 'Year'])['PersonID'].nunique().reset_index()
indep_num.rename(columns={'PersonID': 'IndepNum'}, inplace=True)

# 合并 BoardSize 与 IndepNum
board_df = pd.merge(board_size, indep_num, on=['Stkcd', 'Year'], how='left')
board_df['IndepNum'] = board_df['IndepNum'].fillna(0).astype(int)

# 步骤9：计算独立董事比例
board_df['IndepRatio'] = board_df['IndepNum'] / board_df['BoardSize']
board_df.loc[board_df['BoardSize'] == 0, 'IndepRatio'] = np.nan

# 步骤10：最终董事会特征表列
board_final = board_df[['Stkcd', 'Year', 'BoardSize', 'IndepRatio']].copy()

# ==================== 第三阶段：识别 CEO ====================

# 步骤11：筛选 CEO 候选人
# GTAPosition 包含 "首席执行官" 或 "总经理" 或 "总裁"（排除"副总经理"）
ceo_keywords = ['首席执行官', '总经理', '总裁']
ceo_mask = df['GTAPosition_lower'].apply(lambda x: any(kw in x for kw in ceo_keywords))
# 排除包含"副"的（如副总经理、副总裁）
exclude_vice = ~df['GTAPosition_lower'].str.contains('副', na=False)
ceo_candidates = df[ceo_mask & exclude_vice].copy()

# 步骤12：去重（优先级：Position 含"首席执行官"优先，再按 PersonID 排序）
# 创建排序辅助列：Position 是否为"首席执行官"
ceo_candidates['is_ceo_title'] = ceo_candidates['Position_lower'].str.contains('首席执行官', na=False).astype(int)
# 按 Stkcd, Year, 优先级降序（True优先），PersonID升序
ceo_candidates_sorted = ceo_candidates.sort_values(
    ['Stkcd', 'Year', 'is_ceo_title', 'PersonID'],
    ascending=[True, True, False, True]
)
ceo_unique = ceo_candidates_sorted.drop_duplicates(subset=['Stkcd', 'Year'], keep='first')
ceo_final = ceo_unique[['Stkcd', 'Year', 'PersonID']].copy()
ceo_final.rename(columns={'PersonID': 'PersonID_CEO'}, inplace=True)

# ==================== 第四阶段：识别董事长 ====================

# 步骤14：筛选董事长
chair_mask = df['GTAPosition_lower'] == '董事长'
# 排除副董事长、代理董事长（确保精确匹配）
chair_candidates = df[chair_mask].copy()

# 步骤15：去重
chair_unique = chair_candidates.drop_duplicates(subset=['Stkcd', 'Year'], keep='first')
chair_final = chair_unique[['Stkcd', 'Year', 'PersonID']].copy()
chair_final.rename(columns={'PersonID': 'PersonID_Chair'}, inplace=True)

# ==================== 生成报告 ====================
report_lines = []
report_lines.append("TMT_POSITION 表清洗报告")
report_lines.append("=" * 60)
report_lines.append(f"文件1原始行数: {rows1}")
report_lines.append(f"文件2原始行数: {rows2}")
report_lines.append(f"合并后总行数: {raw_rows}")
report_lines.append(f"筛选年份 (2008-2024) 后行数: {after_year_filter}")
report_lines.append(f"筛选在职状态后行数: {after_status_filter}")
report_lines.append("\n--- 董事会特征表 ---")
report_lines.append(f"观测数: {len(board_final)}")
report_lines.append(f"涉及公司数: {board_final['Stkcd'].nunique()}")
report_lines.append(f"年份范围: {board_final['Year'].min()} - {board_final['Year'].max()}")
report_lines.append("\nBoardSize 描述统计:")
report_lines.append(board_final['BoardSize'].describe().to_string())
report_lines.append("\nIndepRatio 描述统计:")
report_lines.append(board_final['IndepRatio'].describe().to_string())
report_lines.append(f"BoardSize 缺失: {board_final['BoardSize'].isnull().sum()}")
report_lines.append(f"IndepRatio 缺失: {board_final['IndepRatio'].isnull().sum()}")

report_lines.append("\n--- CEO ID 表 ---")
report_lines.append(f"观测数: {len(ceo_final)}")
report_lines.append(f"唯一 CEO 人数: {ceo_final['PersonID_CEO'].nunique()}")
report_lines.append(f"缺失值数量: {ceo_final['PersonID_CEO'].isnull().sum()}")

report_lines.append("\n--- 董事长 ID 表 ---")
report_lines.append(f"观测数: {len(chair_final)}")
report_lines.append(f"唯一董事长人数: {chair_final['PersonID_Chair'].nunique()}")
report_lines.append(f"缺失值数量: {chair_final['PersonID_Chair'].isnull().sum()}")

# 保存报告
report_path = os.path.join(OUTPUT_DIR_REPORT, "TMT_POSITION_report.txt")
with open(report_path, 'w', encoding='utf-8-sig') as f:
    f.write('\n'.join(report_lines))

# ==================== 保存清洗后数据 ====================
board_path = os.path.join(OUTPUT_DIR_CLEANED, "TMT_POSITION_board_cleaned.csv")
board_final.to_csv(board_path, index=False, encoding='utf-8-sig')

ceo_path = os.path.join(OUTPUT_DIR_CLEANED, "TMT_POSITION_ceo_id_cleaned.csv")
ceo_final.to_csv(ceo_path, index=False, encoding='utf-8-sig')

chair_path = os.path.join(OUTPUT_DIR_CLEANED, "TMT_POSITION_chair_id_cleaned.csv")
chair_final.to_csv(chair_path, index=False, encoding='utf-8-sig')

# ==================== 抽样并保存 ====================
sample_board = board_final.sample(n=min(SAMPLE_SIZE, len(board_final)), random_state=SEED)
sample_board_path = os.path.join(OUTPUT_DIR_REPORT, "TMT_POSITION_board_sample.xlsx")
sample_board.to_excel(sample_board_path, index=False)

# ==================== 控制台输出摘要 ====================
print("\n清洗完成！")
print(f"董事会特征表已保存至: {board_path}")
print(f"CEO ID 表已保存至: {ceo_path}")
print(f"董事长 ID 表已保存至: {chair_path}")
print(f"报告已保存至: {report_path}")
print(f"抽样数据已保存至: {sample_board_path}")
print("\n--- 报告摘要 ---")
print(f"董事会特征观测数: {len(board_final)}")
print(f"CEO ID 观测数: {len(ceo_final)}")
print(f"董事长 ID 观测数: {len(chair_final)}")