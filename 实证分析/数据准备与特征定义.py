import pandas as pd
import numpy as np
import os

# ==================== 配置路径 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据处理结果\第三次处理-补充公司信息\更新后的控制变量表\final_data_updated.feather"
OUTPUT_ROOT = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "0_数据准备与特征定义")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局随机种子
SEED = 42
np.random.seed(SEED)

# ==================== 读取数据 ====================
print("读取最终数据...")
df = pd.read_feather(INPUT_FILE)

# ==================== 定义变量组 ====================
control_vars = [
    'Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
    'BoardSize', 'IndepRatio', 'LnFirmAge'
]

ceo_vars = [
    'Female_CEO', 'LnAge_CEO', 'ShareRatio_CEO', 'Duality_CEO',
    'Parttime_CEO', 'ProFun_CEO', 'MgtFun_CEO', 'SkiFun_CEO',
    'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO'
]

chair_vars = [
    'Female_Chair', 'LnAge_Chair', 'ShareRatio_Chair', 'Duality_Chair',
    'Parttime_Chair', 'ProFun_Chair', 'MgtFun_Chair', 'SkiFun_Chair',
    'Oversea_Chair', 'AcademicExp_Chair', 'FinBackExp_Chair'
]

# 校验变量名是否存在
all_vars = control_vars + ceo_vars + chair_vars + ['Stkcd', 'Year', 'IndustryCode', 'NPro', 'STSelf']
missing_vars = [v for v in all_vars if v not in df.columns]
if missing_vars:
    print(f"警告：下列变量不存在于数据中: {missing_vars}")
    # 对于缺失的变量，从列表中移除，避免后续报错
    control_vars = [v for v in control_vars if v in df.columns]
    ceo_vars = [v for v in ceo_vars if v in df.columns]
    chair_vars = [v for v in chair_vars if v in df.columns]

# ==================== 定义6个模型的特征集 ====================
models = {
    "NPro_基准": control_vars,
    "NPro_CEO": control_vars + ceo_vars,
    "NPro_董事长": control_vars + chair_vars,
    "STSelf_基准": control_vars,
    "STSelf_CEO": control_vars + ceo_vars,
    "STSelf_董事长": control_vars + chair_vars,
}

# ==================== 时间窗口定义 ====================
time_windows = {
    "NPro": {
        "train_start": 2015, "train_end": 2023,
        "test_start": 2016, "test_end": 2024   # 训练年+1
    },
    "STSelf": {
        "train_start": 2008, "train_end": 2023,
        "test_start": 2009, "test_end": 2024
    }
}

# 计算各模型训练/测试年及轮数
time_details = {}
for target, window in time_windows.items():
    train_years = list(range(window["train_start"], window["train_end"] + 1))
    test_years = list(range(window["test_start"], window["test_end"] + 1))
    rounds = len(train_years)
    time_details[target] = {
        "训练年份": train_years,
        "测试年份": test_years,
        "轮数": rounds
    }

# ==================== 生成报告1：数据基本统计 ====================
stats_lines = []
stats_lines.append("数据基本统计")
stats_lines.append("=" * 50)
stats_lines.append(f"观测总数: {len(df)}")
stats_lines.append(f"公司数: {df['Stkcd'].nunique() if 'Stkcd' in df.columns else '未知'}")
year_min = int(df['Year'].min()) if 'Year' in df.columns else '未知'
year_max = int(df['Year'].max()) if 'Year' in df.columns else '未知'
stats_lines.append(f"年份范围: {year_min} - {year_max}")
stats_lines.append("")
stats_lines.append("各变量缺失值数量:")
for var in all_vars:
    if var in df.columns:
        miss = df[var].isnull().sum()
        stats_lines.append(f"  {var}: {miss}")
    else:
        stats_lines.append(f"  {var}: 变量不存在")

# 保存
with open(os.path.join(OUTPUT_DIR, "数据基本统计.txt"), 'w', encoding='utf-8') as f:
    f.write('\n'.join(stats_lines))

# ==================== 生成报告2：特征列表 ====================
feat_lines = []
feat_lines.append("模型特征列表")
feat_lines.append("=" * 50)
for model_name, feat_list in models.items():
    feat_lines.append(f"\n【{model_name}】")
    feat_lines.append(f"特征数量: {len(feat_list)}")
    feat_lines.append("特征列表:")
    for i, feat in enumerate(feat_list, 1):
        feat_lines.append(f"  {i}. {feat}")

with open(os.path.join(OUTPUT_DIR, "特征列表.txt"), 'w', encoding='utf-8') as f:
    f.write('\n'.join(feat_lines))

# ==================== 生成报告3：时间窗口划分 ====================
time_lines = []
time_lines.append("时间窗口划分")
time_lines.append("=" * 50)
for target, detail in time_details.items():
    time_lines.append(f"\n【{target}模型】")
    time_lines.append(f"训练年份: {detail['训练年份'][0]} - {detail['训练年份'][-1]} (共 {len(detail['训练年份'])} 年)")
    time_lines.append(f"测试年份: {detail['测试年份'][0]} - {detail['测试年份'][-1]} (共 {len(detail['测试年份'])} 年)")
    time_lines.append(f"滚动窗口轮数: {detail['轮数']} 轮 (训练年 → 训练年+1)")
    time_lines.append(f"详细训练年 → 测试年:")
    for t, t1 in zip(detail['训练年份'], detail['测试年份']):
        time_lines.append(f"  {t} → {t1}")

with open(os.path.join(OUTPUT_DIR, "时间窗口划分.txt"), 'w', encoding='utf-8') as f:
    f.write('\n'.join(time_lines))

print(f"任务完成！输出文件位于: {OUTPUT_DIR}")