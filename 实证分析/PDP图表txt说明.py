import os

output_dir = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果\图表\PDP"
os.makedirs(output_dir, exist_ok=True)

content = """PDP 部分依赖图说明
=====================
图表来源：基于最终数据（final_data_updated.feather）
模型配置：GradientBoostingRegressor，随机种子 42
训练年份：NPro 模型使用 2023 年数据，STSelf 模型使用 2023 年数据
超参数调优：5折交叉验证网格搜索（learning_rate ∈ {0.001, 0.01, 0.1}，max_depth ∈ {2, 4, 6}），早停轮数 50，内部验证比例 0.2
绘制特征：持股比例（原始小数已转换为百分号显示）和年龄对数

横轴含义：
  - 持股比例（%）：ShareRatio_CEO/Chair，取值范围 0~100，反映高管年末持股数量占总股本的比例
  - 年龄（对数）：LnAge_CEO/Chair，反映高管年龄的自然对数值
纵轴含义：部分依赖函数值，表示在控制其他特征不变时，目标变量（NPro 或 STSelf）随该特征变化的平均预测趋势

观察要点：
  - 若曲线呈上升趋势，说明该特征增加与更高的创新指数相关
  - 若曲线平坦，说明该特征对预测的边际贡献较小
  - 不同模型（CEO vs 董事长）的曲线形态可比较其影响差异

注意：部分依赖图反映的是平均边际效应，不代表因果关系。
"""

with open(os.path.join(output_dir, "PDP图表说明.txt"), "w", encoding="utf-8") as f:
    f.write(content)

print("PDP图表说明.txt 已生成。")