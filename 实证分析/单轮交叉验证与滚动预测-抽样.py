import pandas as pd
import os

# 文件路径
cv_csv = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果\NPro_CEO\交叉验证\2015_CV结果.csv"
pred_csv = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果\NPro_CEO\滚动预测\2015_预测.csv"

SAMPLE_SIZE = 50
SEED = 42

def sample_and_save(filepath):
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    df = pd.read_csv(filepath)
    n = min(SAMPLE_SIZE, len(df))
    df_sample = df.sample(n=n, random_state=SEED)
    # 生成抽样文件名：原文件名（不含扩展名） + _sample.csv
    base, ext = os.path.splitext(filepath)
    out_path = f"{base}_sample.csv"
    df_sample.to_csv(out_path, index=False)
    print(f"已保存: {out_path} (抽样 {n} 条)")

sample_and_save(cv_csv)
sample_and_save(pred_csv)
print("抽样完成。")