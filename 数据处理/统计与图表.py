import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# ==================== 配置路径 ====================
FINAL_DATA = r"D:\how dare you\2026统计建模\数据处理\第三次处理-补充公司信息\更新后的控制变量表\final_data_updated.csv"
WINSOR_FILE = r"D:\how dare you\2026统计建模\数据处理\第三次处理-最终数据\报告与抽样\winsorization_summary.xlsx"  # 可能存在的缩尾对比表
OUTPUT_DIR = r"D:\how dare you\2026统计建模\数据处理\统计报告与图片"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 学术绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 读取数据 ====================
print("读取最终数据...")
df = pd.read_csv(FINAL_DATA, dtype={'Stkcd': str, 'IndustryCode': str})

# 统一年份为整数
if 'Year' in df.columns:
    df['Year'] = df['Year'].astype(int)

# 生成行业门类（取首字母）
df['IndustrySector'] = df['IndustryCode'].fillna('').astype(str).str[0]
df['IndustrySector'] = df['IndustrySector'].replace('', '缺失')

# ==================== 1. 样本年度–行业分布 ====================
print("生成样本分布统计...")

# 统计表格
industry_year = df.groupby(['Year', 'IndustrySector']).size().unstack(fill_value=0)
industry_year['合计'] = industry_year.sum(axis=1)
industry_year.loc['合计'] = industry_year.sum(axis=0)

# 占比（各年内部比例）
industry_year_pct = industry_year.div(industry_year['合计'], axis=0).drop('合计', axis=0).fillna(0) * 100

# 保存文本表格
dist_text_path = os.path.join(OUTPUT_DIR, "样本分布_industry_year.txt")
with open(dist_text_path, 'w', encoding='utf-8') as f:
    f.write("样本年度–行业分布（观测数）\n")
    f.write("=" * 60 + "\n")
    f.write(industry_year.to_string())
    f.write("\n\n样本年度–行业分布（列百分比，%）\n")
    f.write("=" * 60 + "\n")
    f.write(industry_year_pct.round(2).to_string())
print(f"  已保存: {dist_text_path}")

# 绘制分组柱状图
plot_data = df.groupby(['Year', 'IndustrySector']).size().reset_index(name='Count')
plt.figure(figsize=(14, 6))
sns.barplot(data=plot_data, x='Year', y='Count', hue='IndustrySector', palette='Set2')
plt.title('样本年度–行业分布', fontsize=14)
plt.xlabel('年份')
plt.ylabel('观测数')
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='行业门类', loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "样本分布_industry_year.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {plot_path}")

# ==================== 2. 描述性统计 ====================
print("生成描述性统计...")

# 定义变量清单
continuous_vars = [
    'NPro', 'STSelf', 'Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
    'BoardSize', 'IndepRatio', 'LnFirmAge', 'LnAge_CEO', 'ShareRatio_CEO',
    'LnAge_Chair', 'ShareRatio_Chair'
]
dummy_vars = [
    'Female_CEO', 'Duality_CEO', 'Parttime_CEO', 'ProFun_CEO', 'MgtFun_CEO',
    'SkiFun_CEO', 'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO',
    'Female_Chair', 'Duality_Chair', 'Parttime_Chair', 'ProFun_Chair',
    'MgtFun_Chair', 'SkiFun_Chair', 'Oversea_Chair', 'AcademicExp_Chair', 'FinBackExp_Chair'
]

all_vars = continuous_vars + dummy_vars
existing_vars = [v for v in all_vars if v in df.columns]

desc_df = df[existing_vars].describe(percentiles=[.5]).T
desc_df = desc_df[['count', 'mean', 'std', 'min', '50%', 'max']]
desc_df.columns = ['观测数', '均值', '标准差', '最小值', '中位数', '最大值']
desc_df = desc_df.round(4)

desc_path = os.path.join(OUTPUT_DIR, "描述性统计_summary.txt")
with open(desc_path, 'w', encoding='utf-8') as f:
    f.write("变量描述性统计\n")
    f.write("=" * 60 + "\n")
    f.write(desc_df.to_string())
print(f"  已保存: {desc_path}")

# ==================== 3. Pearson 相关系数矩阵 ====================
print("计算 Pearson 相关系数矩阵（含显著性星号）...")

cont_existing = [v for v in continuous_vars if v in df.columns]
corr_matrix = pd.DataFrame(index=cont_existing, columns=cont_existing, dtype=str)

for i, var1 in enumerate(cont_existing):
    for j, var2 in enumerate(cont_existing):
        if i > j:  # 下三角
            sub = df[[var1, var2]].dropna()
            if len(sub) > 2:
                r, p = pearsonr(sub[var1], sub[var2])
                stars = ''
                if p < 0.001: stars = '***'
                elif p < 0.01: stars = '**'
                elif p < 0.05: stars = '*'
                corr_matrix.iloc[i, j] = f"{r:.3f}{stars}"
            else:
                corr_matrix.iloc[i, j] = 'NaN'
        elif i == j:
            corr_matrix.iloc[i, j] = '1.000'
        else:
            corr_matrix.iloc[i, j] = ''

corr_path = os.path.join(OUTPUT_DIR, "相关系数矩阵_pearson.txt")
with open(corr_path, 'w', encoding='utf-8') as f:
    f.write("Pearson 相关系数矩阵（下三角，* p<0.05, ** p<0.01, *** p<0.001）\n")
    f.write("=" * 80 + "\n")
    f.write(corr_matrix.to_string())
print(f"  已保存: {corr_path}")

# ==================== 4. 缩尾前后对比 ====================
print("处理缩尾对比...")
winsor_path_out = os.path.join(OUTPUT_DIR, "缩尾对比_winsorization.txt")

if os.path.exists(WINSOR_FILE):
    df_winsor = pd.read_excel(WINSOR_FILE)
    with open(winsor_path_out, 'w', encoding='utf-8') as f:
        f.write("缩尾前后描述统计对比（基于 1% 和 99% 分位数）\n")
        f.write("=" * 80 + "\n")
        f.write(df_winsor.to_string(index=False))
    print(f"  已从既有文件生成: {winsor_path_out}")
else:
    # 若文件不存在，生成简化版
    winsor_cols = [c for c in continuous_vars if c in df.columns]
    summary_before = df[winsor_cols].describe().T[['min', 'max', 'mean', 'std']]
    # 模拟缩尾后（实际已经缩尾，此处仅示例）
    with open(winsor_path_out, 'w', encoding='utf-8') as f:
        f.write("缩尾对比文件未找到，请参考前期报告。\n")
        f.write("当前最终数据已为缩尾后版本。\n")
    print(f"  警告: 缩尾对比文件不存在，已生成占位说明: {winsor_path_out}")

# ==================== 5. 缺失率分析 ====================
print("生成缺失率统计...")
missing_final = df.isnull().sum()
missing_final_pct = (missing_final / len(df)) * 100
missing_df = pd.DataFrame({'缺失数': missing_final, '缺失率(%)': missing_final_pct})
missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失数', ascending=False)

missing_path = os.path.join(OUTPUT_DIR, "缺失率_missing_rate.txt")
with open(missing_path, 'w', encoding='utf-8') as f:
    f.write("最终数据变量缺失情况\n")
    f.write("=" * 50 + "\n")
    if len(missing_df) == 0:
        f.write("所有变量均无缺失。\n")
    else:
        f.write(missing_df.to_string())
    f.write("\n\n注：前期各表缺失率请参考构造表阶段报告。\n")
print(f"  已保存: {missing_path}")

# ==================== 6. 方差膨胀因子（VIF） ====================
print("计算 VIF...")

# 选择自变量：控制变量 + 核心高管特征（排除被解释变量和ID）
exclude = ['Stkcd', 'Year', 'IndustryCode', 'IndustrySector', 'NPro', 'STSelf',
           'TotalShares', 'LnFirmAge', 'StateShare', 'Top10HoldRatio',
           'ShareRatio_CEO', 'ShareRatio_Chair']
# 构建自变量列表
X_vars = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
# 手动添加关键变量确保覆盖
core_vars = ['Size', 'Leverage', 'PPE', 'BoardSize', 'IndepRatio',
             'LnAge_CEO', 'LnAge_Chair',
             'Female_CEO', 'Duality_CEO', 'Parttime_CEO', 'ProFun_CEO', 'MgtFun_CEO', 'SkiFun_CEO',
             'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO',
             'Female_Chair', 'Duality_Chair', 'Parttime_Chair', 'ProFun_Chair', 'MgtFun_Chair', 'SkiFun_Chair',
             'Oversea_Chair', 'AcademicExp_Chair', 'FinBackExp_Chair']
X_vars = [v for v in core_vars if v in df.columns]

# 剔除缺失值
X = df[X_vars].dropna()
if X.shape[0] > 10 and X.shape[1] > 1:
    vif_data = pd.DataFrame()
    vif_data['变量'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
else:
    vif_data = pd.DataFrame({'变量': [], 'VIF': []})

vif_path = os.path.join(OUTPUT_DIR, "共线性诊断_vif.txt")
with open(vif_path, 'w', encoding='utf-8') as f:
    f.write("方差膨胀因子（VIF）检验结果\n")
    f.write("=" * 40 + "\n")
    f.write("（VIF > 10 提示严重共线性）\n\n")
    if vif_data.empty:
        f.write("无法计算 VIF（样本不足或变量缺失）。\n")
    else:
        f.write(vif_data.to_string(index=False))
print(f"  已保存: {vif_path}")

# ==================== 7. 衍生变量构成说明 ====================
print("生成衍生变量构成说明...")
explain_text = """衍生变量构成说明
================

1. 新质生产力指数（NPro）
   由以下11个三级指标通过年度熵值法合成：
   - RDPSalaryRatio     研发人员薪资占比
   - RDPersonRatio      研发人员占比
   - HighEduPersonRatio 高学历人员占比
   - FixedAssetsRatio   固定资产占比
   - ManufacturCostsRatio 制造费用占比
   - RDPDepAmortRatio   研发折旧摊销占比
   - RDPLeaseCostsRatio 研发租赁费占比
   - DirectInvestment   研发直接投入占比
   - IntangibleAssetsRatio 无形资产占比
   - AssetTurnover      总资产周转率
   - EquityMultiplierRec 权益乘数倒数
   来源表：NQPF_EnNQPThreeLevelIndLT

2. 科技自立自强指数（STSelf）
   由以下5个二级指标通过年度熵值法合成：
   - RDRatio           研发投入强度（研发支出/营业收入）
   - RDPersonRatio     研发人员占比
   - LnInvPatGrant     发明专利授权数对数 ln(1+授权数)
   - InvPatGrantRatio  发明专利授权率（授权数/申请数）
   - LnValidInvPat     有效发明专利存量对数 ln(1+存量)
   来源表：PT_LCRDSPENDING、TIRD_EntRDTecCap、PT_LCDETAIL、PT_VALID

3. 高管背景哑变量（CEO/董事长分别计算）
   基于 TMT_FIGUREINFO 表构造，规则如下：
   - Female：性别为女=1
   - LnAge：ln(年龄)
   - ShareRatio：年末持股数 / 总股本
   - Duality：是否两职合一（来自 IsDuality 字段）
   - Parttime：是否外部兼职（IsCocurP=1 或 Director_ListCO 非空非0）
   - ProFun：是否生产/研发/设计背景（Funback 含1/2/3）
   - MgtFun：是否管理/市场/人力背景（Funback 含4/5/6）
   - SkiFun：是否财务/法律背景（Funback 含8/9）
   - Oversea：是否有海外背景（OveseaBack 非空且不含3）
   - AcademicExp：是否有学术背景（Academic 非空且非4）
   - FinBackExp：是否有金融背景（FinBack 非空且非99）
   详细逻辑见流程文档 5.2 节。
"""
explain_path = os.path.join(OUTPUT_DIR, "衍生变量构成说明.txt")
with open(explain_path, 'w', encoding='utf-8') as f:
    f.write(explain_text)
print(f"  已保存: {explain_path}")

print("\n所有统计报告与图表生成完成！")
print(f"输出目录: {OUTPUT_DIR}")