import pandas as pd
import os
import warnings

# 屏蔽 openpyxl 读取 Excel 时的默认样式警告（不影响数据内容）
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# 设置随机种子
SEED = 42
SAMPLE_SIZE = 20

# 输出目录
output_dir = r"D:\how dare you\2026统计建模\数据处理\原始数据抽样"
os.makedirs(output_dir, exist_ok=True)

# 源文件清单及输出文件名
files = [
    (r"D:\how dare you\2026统计建模\数据\资产负债表122033492(仅供UC Berkeley使用)\FS_Combas.xlsx", "FS_Combas_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\股本结构\股本结构.xlsx", "股本结构_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\大股东持股\大股东持股_1.xlsx", "大股东持股_1_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\大股东持股\大股东持股_2.xlsx", "大股东持股_2_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\上市公司基本信息163004282(仅供北洋大学使用)\CSp_ListedCoInfoAnl.xlsx", "CSp_ListedCoInfoAnl_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\董监高任职情况表103859776(仅供UC Berkeley使用)\TMT_POSITION.xlsx", "TMT_POSITION_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\董监高任职情况表103859776(仅供UC Berkeley使用)\TMT_POSITION1.xlsx", "TMT_POSITION1_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\董监高个人特征文件162730779(仅供北洋大学使用)\TMT_FIGUREINFO.xlsx", "TMT_FIGUREINFO_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\董监高个人特征文件162730779(仅供北洋大学使用)\TMT_FIGUREINFO1.xlsx", "TMT_FIGUREINFO1_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\企业新质生产力三级指标表(劳动力、生产工具角度)163242606(仅供北洋大学使用)\NQPF_EnNQPThreeLevelIndLT.xlsx", "NQPF_EnNQPThreeLevelIndLT_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\研发投入情况表115449623(仅供UC Berkeley使用)\PT_LCRDSPENDING.xlsx", "PT_LCRDSPENDING_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\企业研发技术能力情况表101050028(仅供UC Berkeley使用)\TIRD_EntRDTecCap.xlsx", "TIRD_EntRDTecCap_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\专利明细情况115313220(仅供UC Berkeley使用)\PT_LCDETAIL.xlsx", "PT_LCDETAIL_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\专利有效情况表(1990-2017)121327553(仅供UC Berkeley使用)\PT_VALID.xlsx", "PT_VALID_sample.xlsx"),
    (r"D:\how dare you\2026统计建模\数据\利润表122235767(仅供UC Berkeley使用)\FS_Comins.xlsx", "FS_Comins_sample.xlsx"),
]

for src_path, out_name in files:
    try:
        df = pd.read_excel(src_path)
        n = min(SAMPLE_SIZE, len(df))
        df_sample = df.sample(n=n, random_state=SEED) if n > 0 else df
        out_path = os.path.join(output_dir, out_name)
        df_sample.to_excel(out_path, index=False)
        print(f"已完成: {out_name}，抽取 {n} 行")
    except Exception as e:
        print(f"处理文件失败: {src_path}\n错误信息: {e}")

print("\n全部抽样完成，请检查输出目录。")