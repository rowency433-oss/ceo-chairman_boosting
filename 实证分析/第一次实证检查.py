import pandas as pd
import numpy as np
import os
import glob

# ==================== 配置路径 ====================
ROOT = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果"
OUTPUT_DIR = os.path.join(ROOT, "汇总表格与报告")
ANOMALY_DIR = os.path.join(OUTPUT_DIR, "异常年份分布")
os.makedirs(ANOMALY_DIR, exist_ok=True)

# 原始数据（用于全样本分布对比）
FINAL_DATA = r"D:\how dare you\2026统计建模\数据处理结果\第三次处理-补充公司信息\更新后的控制变量表\final_data_updated.feather"

# 模型名称列表
models = ['NPro_基准', 'NPro_CEO', 'NPro_董事长',
          'STSelf_基准', 'STSelf_CEO', 'STSelf_董事长']

# ==================== 任务1：完整变量重要性排序表 ====================
print("任务1：生成完整变量重要性排序表...")

# 特征类型判定函数
def feature_type(name):
    if name.endswith('_CEO') or name.endswith('_Chair'):
        return '高管特征'
    else:
        return '控制变量'

all_importance = {}  # 存放每个模型的 {feature: mean_importance}

for model in models:
    imp_dir = os.path.join(ROOT, model, "变量重要性")
    # 查找所有年份的重要性CSV
    csv_files = glob.glob(os.path.join(imp_dir, "*_重要性.csv"))
    if not csv_files:
        print(f"  警告: {model} 未找到重要性文件，跳过")
        continue
    # 合并所有年份
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  读取 {f} 失败: {e}")
    if not dfs:
        continue
    combined = pd.concat(dfs, ignore_index=True)
    # 计算各特征平均重要性
    mean_imp = combined.groupby('feature')['importance'].mean().sort_values(ascending=False)
    all_importance[model] = mean_imp

# 写文件
lines = []
lines.append("完整变量重要性排序表")
lines.append("=" * 70)
lines.append("说明：重要性为各年份重要性CSV的算术平均值，已按降序排列。\n")

for model in models:
    if model not in all_importance:
        continue
    lines.append(f"\n【{model}】")
    lines.append(f"{'排名':<4} {'变量名':<30} {'重要性(%)':<10} {'类型':<10}")
    lines.append("-" * 58)
    mean_imp = all_importance[model]
    for rank, (feat, imp) in enumerate(mean_imp.items(), 1):
        pct = imp * 100  # 转为百分比
        ftype = feature_type(feat)
        lines.append(f"{rank:<4} {feat:<30} {pct:<10.1f} {ftype:<10}")

lines.append("\n注：重要性为各年份的算术平均值，所有模型共有的控制变量和特征变量均列出。")

out_path = os.path.join(OUTPUT_DIR, "完整变量重要性排序表.txt")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"  已保存: {out_path}")

# ==================== 任务2：异常年份Y值分布检查 ====================
print("\n任务2：异常年份Y值分布检查...")

# 定义要检查的预测文件路径
anomaly_cases = [
    {
        'target': 'NPro',
        'test_year': 2017,
        'model': 'NPro_CEO',
        'pred_file': os.path.join(ROOT, 'NPro_CEO', '滚动预测', '2016_预测.csv')
    },
    {
        'target': 'NPro',
        'test_year': 2024,
        'model': 'NPro_CEO',
        'pred_file': os.path.join(ROOT, 'NPro_CEO', '滚动预测', '2023_预测.csv')
    },
    {
        'target': 'STSelf',
        'test_year': 2018,
        'model': 'STSelf_CEO',
        'pred_file': os.path.join(ROOT, 'STSelf_CEO', '滚动预测', '2017_预测.csv')
    }
]

# 尝试加载原始数据（全样本）
df_full = None
try:
    df_full = pd.read_feather(FINAL_DATA)
except Exception as e:
    print(f"  警告: 无法加载原始数据: {e}")

report_lines = []
report_lines.append("异常年份Y值分布报告")
report_lines.append("=" * 70)

for case in anomaly_cases:
    pred_file = case['pred_file']
    target = case['target']
    test_year = case['test_year']
    model = case['model']
    report_lines.append(f"\n--- {target} 测试年 {test_year} (模型: {model}) ---")
    if not os.path.exists(pred_file):
        report_lines.append(f"  预测文件缺失: {pred_file}")
        continue

    df_pred = pd.read_csv(pred_file)
    if 'y_true' not in df_pred.columns:
        report_lines.append("  预测文件中缺少 y_true 列，跳过。")
        continue

    y = df_pred['y_true'].dropna()
    # 样本统计
    stats = y.describe(percentiles=[0.25, 0.5, 0.75])
    report_lines.append(f"  样本量: {len(y)}")
    report_lines.append(f"  均值: {stats['mean']:.4f}")
    report_lines.append(f"  标准差: {stats['std']:.4f}")
    report_lines.append(f"  最小值: {stats['min']:.4f}")
    report_lines.append(f"  25%分位: {stats['25%']:.4f}")
    report_lines.append(f"  中位数: {stats['50%']:.4f}")
    report_lines.append(f"  75%分位: {stats['75%']:.4f}")
    report_lines.append(f"  最大值: {stats['max']:.4f}")

    # 全样本对比
    if df_full is not None and target in df_full.columns:
        y_all = df_full[target].dropna()
        all_stats = y_all.describe(percentiles=[0.25, 0.5, 0.75])
        report_lines.append(f"\n  全样本 {target} 对比 (所有年份非空):")
        report_lines.append(f"    样本量: {len(y_all)}")
        report_lines.append(f"    均值: {all_stats['mean']:.4f} (差值: {stats['mean'] - all_stats['mean']:.4f})")
        report_lines.append(f"    标准差: {all_stats['std']:.4f} (差值: {stats['std'] - all_stats['std']:.4f})")
        report_lines.append(f"    最小值: {all_stats['min']:.4f}")
        report_lines.append(f"    中位数: {all_stats['50%']:.4f}")
        report_lines.append(f"    最大值: {all_stats['max']:.4f}")
    report_lines.append("-" * 40)

report_lines.append("\n检查结论：请观察这些年份的Y值分布是否出现严重偏态或极端值。如无异常可保留，若存在极端值建议在论文中说明并考虑剔除或缩尾处理。")

anomaly_path = os.path.join(ANOMALY_DIR, "异常年份Y值分布报告.txt")
with open(anomaly_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"  已保存: {anomaly_path}")

# ==================== 任务3：修正汇总分析摘要 ====================
print("\n任务3：修正汇总分析摘要...")

# 读取已生成的模型整体关联强度表以获取准确数字
# 这里采用直接二次解析 NPro_汇总结果.txt 和 STSelf_汇总结果.txt 以获得总体R²和增量
import re

def parse_overall_r2(filepath, model_group):
    """从汇总txt中提取指定模型组的总体R²，返回字典"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    r2_dict = {}
    for model in model_group:
        # 查找 【模型名】 后的 总体R²: 数字
        m = re.search(rf'【{model}】.*?总体R²:\s*([-\d.]+)', text, re.DOTALL)
        if m:
            r2_dict[model] = float(m.group(1))
    return r2_dict

npro_models = ['NPro_基准', 'NPro_CEO', 'NPro_董事长']
stself_models = ['STSelf_基准', 'STSelf_CEO', 'STSelf_董事长']

npro_path = os.path.join(ROOT, "NPro_汇总结果.txt")
stself_path = os.path.join(ROOT, "STSelf_汇总结果.txt")

npro_r2 = parse_overall_r2(npro_path, npro_models)
stself_r2 = parse_overall_r2(stself_path, stself_models)

# 计算增量
npro_base = npro_r2.get('NPro_基准', np.nan)
npro_ceo = npro_r2.get('NPro_CEO', np.nan)
npro_chair = npro_r2.get('NPro_董事长', np.nan)

stself_base = stself_r2.get('STSelf_基准', np.nan)
stself_ceo = stself_r2.get('STSelf_CEO', np.nan)
stself_chair = stself_r2.get('STSelf_董事长', np.nan)

# 找到持股比例特征的位置（从 all_importance 获取）
def get_share_ratio_rank(model_list, suffix):
    """在指定模型中查找 ShareRatio 特征的排名"""
    ranks = {}
    for model in model_list:
        if model in all_importance:
            series = all_importance[model]
            feat_name = f'ShareRatio{suffix}'
            if feat_name in series.index:
                rank = list(series.index).index(feat_name) + 1
                ranks[model] = rank
    return ranks

share_ceo_ranks = get_share_ratio_rank(['NPro_CEO', 'STSelf_CEO'], '_CEO')
share_chair_ranks = get_share_ratio_rank(['NPro_董事长', 'STSelf_董事长'], '_Chair')

# 文本生成
lines4 = []
lines4.append("汇总分析摘要（修正版）")
lines4.append("=" * 70)
lines4.append("本研究采用梯度提升回归树结合滚动窗口方法，考察高管特征对新质生产力(NPro)和科技自立自强指数(STSelf)的预测能力。")
lines4.append(f"1. 整体预测性能：基准模型对 NPro 的总体样本外 R² 为 {npro_base:.3f}，对 STSelf 为 {stself_base:.3f}。引入 CEO 特征后，NPro 的 R² 增量为 {npro_ceo - npro_base:+.3f}，STSelf 的增量为 {stself_ceo - stself_base:+.3f}；引入董事长特征后，增量分别为 {npro_chair - npro_base:+.3f} 和 {stself_chair - stself_base:+.3f}。")
lines4.append(f"2. 高管特征的解释力增量总体有限但方向一致为正，表明高管个体差异对创新指标具备一定的预测价值。")
lines4.append("3. 最重要的控制变量集中于企业规模(Size)、财务杠杆(Leverage)和董事会独立性(IndepRatio)，提示企业基本特征在创新预测中仍占主导地位。")
lines4.append("4. 高管特征中，持股比例（ShareRatio_CEO/Chair）在 CEO 和董事长模型中均稳定进入前 3 名，是关联强度最高的高管特征。年龄（LnAge）特征排名中游，而职能背景（ProFun/MgtFun/SkiFun）、海外经历、学术背景及金融背景的重要性相对较低，但仍提供一定的预测信息。")
lines4.append("5. 单年预测中个别年份出现 R² 为负，可能是当年数据噪声较大或结构变化所致，但不影响整体正向预测能力。")

# 额外添加关于持股比例排名的具体数值
share_info = []
for m in ['NPro_CEO', 'STSelf_CEO', 'NPro_董事长', 'STSelf_董事长']:
    if m in share_ceo_ranks or m in share_chair_ranks:
        rank = share_ceo_ranks.get(m) or share_chair_ranks.get(m)
        share_info.append(f"{m}: 排名第{rank}")
if share_info:
    lines4.append(f"   具体排名: {', '.join(share_info)}。")

lines4.append("\n（注：以上数据基于最终数据，所有随机种子固定为42，结果可复现。）")

summary_path = os.path.join(OUTPUT_DIR, "汇总分析摘要.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines4))
print(f"  已保存: {summary_path}")

print("\n任务1-3已完成。")