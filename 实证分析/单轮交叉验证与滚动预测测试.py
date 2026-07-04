import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 配置路径 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据处理结果\第三次处理-补充公司信息\更新后的控制变量表\final_data_updated.feather"
OUTPUT_ROOT = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果"
MODEL_NAME = "NPro_CEO"
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, MODEL_NAME)
CV_DIR = os.path.join(OUTPUT_DIR, "交叉验证")
PRED_DIR = os.path.join(OUTPUT_DIR, "滚动预测")

os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# 全局随机种子
SEED = 42

# ==================== 定义变量 ====================
control_vars = [
    'Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
    'BoardSize', 'IndepRatio', 'LnFirmAge'
]
ceo_vars = [
    'Female_CEO', 'LnAge_CEO', 'ShareRatio_CEO', 'Duality_CEO',
    'Parttime_CEO', 'ProFun_CEO', 'MgtFun_CEO', 'SkiFun_CEO',
    'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO'
]
features = control_vars + ceo_vars
target = 'NPro'
train_year = 2015
test_year = 2016

# ==================== 数据准备 ====================
print("加载数据...")
df = pd.read_feather(INPUT_FILE)

# 筛选年份并删除目标缺失
df_train = df[(df['Year'] == train_year) & df[target].notna()].copy()
df_test = df[(df['Year'] == test_year) & df[target].notna()].copy()

# 仅保留特征+Stkcd+Year+目标，并删除任何含缺失的行
cols_to_keep = ['Stkcd', 'Year'] + features + [target]
df_train = df_train[cols_to_keep].dropna()
df_test = df_test[cols_to_keep].dropna()

X_train_full = df_train[features]
y_train_full = df_train[target]
X_test = df_test[features]
y_test = df_test[target]
test_stkcd = df_test['Stkcd'].values

print(f"训练集观测数: {len(X_train_full)}")
print(f"测试集观测数: {len(X_test)}")

# ==================== 超参数网格 ====================
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [2, 4, 6]
}
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

# ==================== 手动网格搜索 + 5折CV ====================
cv_results = []

print(f"开始网格搜索（{len(param_grid['learning_rate']) * len(param_grid['max_depth'])} 个组合，{n_folds}折CV）...")

for lr in param_grid['learning_rate']:
    for md in param_grid['max_depth']:
        fold_errors = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
            X_tr = X_train_full.iloc[train_idx]
            y_tr = y_train_full.iloc[train_idx]
            X_val = X_train_full.iloc[val_idx]
            y_val = y_train_full.iloc[val_idx]

            # 创建GradientBoostingRegressor，启用早停
            model = GradientBoostingRegressor(
                learning_rate=lr,
                max_depth=md,
                n_estimators=5000,
                random_state=SEED,
                subsample=1.0,  # 默认值
                validation_fraction=0.1,  # 从训练集中留出10%作为验证集
                n_iter_no_change=50,  # 连续50轮验证分数不提升则停止
                tol=1e-4,
                verbose=0
            )
            model.fit(X_tr, y_tr)
            # 用验证集（该fold的val部分）评估
            y_pred_val = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred_val)
            fold_errors.append(mse)

        mean_mse = np.mean(fold_errors)
        cv_results.append({
            'learning_rate': lr,
            'max_depth': md,
            'fold_1': fold_errors[0],
            'fold_2': fold_errors[1],
            'fold_3': fold_errors[2],
            'fold_4': fold_errors[3],
            'fold_5': fold_errors[4],
            'mean_error': mean_mse
        })
        print(f"  lr={lr}, md={md} → Mean MSE: {mean_mse:.6f}")

# 保存CV结果
df_cv = pd.DataFrame(cv_results)
df_cv.to_csv(os.path.join(CV_DIR, "2015_CV结果.csv"), index=False, encoding='utf-8-sig')
print(f"CV结果已保存至: {os.path.join(CV_DIR, '2015_CV结果.csv')}")

# 选择最优超参数（平均误差最小）
best_row = df_cv.loc[df_cv['mean_error'].idxmin()]
best_lr = best_row['learning_rate']
best_md = int(best_row['max_depth'])
best_cv_error = best_row['mean_error']
print(f"最优超参数: learning_rate={best_lr}, max_depth={best_md}, CV Mean MSE={best_cv_error:.6f}")

# ==================== 训练最终模型 ====================
print("在完整2015训练集上训练最优模型...")
final_model = GradientBoostingRegressor(
    learning_rate=best_lr,
    max_depth=best_md,
    n_estimators=5000,
    random_state=SEED,
    subsample=1.0,
    validation_fraction=0.1,
    n_iter_no_change=50,
    tol=1e-4,
    verbose=0
)
final_model.fit(X_train_full, y_train_full)
actual_estimators = final_model.n_estimators_  # 早停后实际使用的树数量
print(f"实际 n_estimators: {actual_estimators}")

# ==================== 预测2016年 ====================
y_pred = final_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

# 保存预测结果
df_pred = pd.DataFrame({
    'Stkcd': test_stkcd,
    'Year': test_year,
    'y_true': y_test.values,
    'y_pred': y_pred
})
df_pred.to_csv(os.path.join(PRED_DIR, "2015_预测.csv"), index=False, encoding='utf-8-sig')
print(f"预测结果已保存至: {os.path.join(PRED_DIR, '2015_预测.csv')}")

# ==================== 生成摘要 ====================
summary_lines = []
summary_lines.append(f"{MODEL_NAME} 单轮测试摘要 (2015→2016)")
summary_lines.append("=" * 60)
summary_lines.append(f"训练集观测数: {len(X_train_full)}")
summary_lines.append(f"测试集观测数: {len(X_test)}")
summary_lines.append("")
summary_lines.append("最优超参数组合:")
summary_lines.append(f"  learning_rate = {best_lr}")
summary_lines.append(f"  max_depth = {best_md}")
summary_lines.append(f"  early_stopping_rounds = 50 (自动确定 n_estimators = {actual_estimators})")
summary_lines.append("")
summary_lines.append(f"5折交叉验证平均 MSE (最优): {best_cv_error:.6f}")
summary_lines.append("")
summary_lines.append("测试集性能:")
summary_lines.append(f"  RMSE = {test_rmse:.6f}")
summary_lines.append(f"  R²   = {test_r2:.6f}")
summary_lines.append("")
# 边界检查
boundary_flag = False
notes = []
if best_lr == param_grid['learning_rate'][0] or best_lr == param_grid['learning_rate'][-1]:
    boundary_flag = True
    notes.append("学习率位于网格边界，建议扩展网格。")
if best_md == param_grid['max_depth'][0] or best_md == param_grid['max_depth'][-1]:
    boundary_flag = True
    notes.append("max_depth位于网格边界，建议扩展网格。")
if actual_estimators >= 4900:  # 接近5000上限
    notes.append("早停法未显著减少迭代次数（n_estimators接近5000），可适当增大上限或调低学习率。")
if not notes:
    notes.append("无异常。")

summary_lines.append("注意事项:")
for note in notes:
    summary_lines.append(f"  - {note}")

summary_path = os.path.join(OUTPUT_ROOT, f"{MODEL_NAME}_单轮测试摘要.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines))

print(f"摘要已保存至: {summary_path}")
print("\n任务完成。请检查输出文件。")