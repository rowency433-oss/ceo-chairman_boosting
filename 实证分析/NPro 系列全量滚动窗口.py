import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 配置 ====================
INPUT_FILE = r"D:\how dare you\2026统计建模\数据处理结果\第三次处理-补充公司信息\更新后的控制变量表\final_data_updated.feather"
ROOT = r"D:\how dare you\2026统计建模\数据处理结果\实证分析结果"
SEED = 42
TRAIN_YEARS = list(range(2015, 2024))  # 2015~2023
TEST_YEARS = [y + 1 for y in TRAIN_YEARS]  # 2016~2024
TARGET = 'NPro'

# 特征组
base_features = ['Size', 'Leverage', 'PPE', 'StateShare', 'Top10HoldRatio',
                 'BoardSize', 'IndepRatio', 'LnFirmAge']
ceo_features = base_features + ['Female_CEO', 'LnAge_CEO', 'ShareRatio_CEO', 'Duality_CEO',
                                'Parttime_CEO', 'ProFun_CEO', 'MgtFun_CEO', 'SkiFun_CEO',
                                'Oversea_CEO', 'AcademicExp_CEO', 'FinBackExp_CEO']
chair_features = base_features + ['Female_Chair', 'LnAge_Chair', 'ShareRatio_Chair', 'Duality_Chair',
                                  'Parttime_Chair', 'ProFun_Chair', 'MgtFun_Chair', 'SkiFun_Chair',
                                  'Oversea_Chair', 'AcademicExp_Chair', 'FinBackExp_Chair']

model_configs = {
    'NPro_基准': base_features,
    'NPro_CEO': ceo_features,
    'NPro_董事长': chair_features,
}

# 超参数网格
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [2, 4, 6]
}

# ==================== 载入数据 ====================
print("加载数据...")
df = pd.read_feather(INPUT_FILE)

# ==================== 工具函数 ====================
def prepare_data(df, features, target, year):
    """提取指定年份的训练集或测试集，删除特征或目标缺失的行"""
    data = df[df['Year'] == year].copy()
    # 必须保留 Stkcd 和 Year 用于后续输出
    subset = data[['Stkcd', 'Year'] + features + [target]].dropna()
    X = subset[features]
    y = subset[target]
    return X, y, subset[['Stkcd', 'Year']]

def perform_cv(X_train, y_train):
    """5折CV遍历所有参数组合，返回CV结果DataFrame和最优参数"""
    cv_results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for lr in param_grid['learning_rate']:
        for md in param_grid['max_depth']:
            fold_mse = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]
                model = GradientBoostingRegressor(
                    learning_rate=lr, max_depth=md,
                    n_estimators=5000, random_state=SEED,
                    validation_fraction=0.2,
                    n_iter_no_change=50, tol=1e-4
                )
                model.fit(X_tr, y_tr)
                y_pred_val = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred_val)
                fold_mse.append(mse)
            mean_mse = np.mean(fold_mse)
            cv_results.append({
                'learning_rate': lr, 'max_depth': md,
                'fold_1': fold_mse[0], 'fold_2': fold_mse[1],
                'fold_3': fold_mse[2], 'fold_4': fold_mse[3],
                'fold_5': fold_mse[4], 'mean_error': mean_mse
            })
    cv_df = pd.DataFrame(cv_results)
    best_row = cv_df.loc[cv_df['mean_error'].idxmin()]
    best_params = {
        'learning_rate': best_row['learning_rate'],
        'max_depth': int(best_row['max_depth'])
    }
    return cv_df, best_params

def train_final_model(X_train, y_train, best_params):
    """用最优参数在完整训练集上训练（使用早停）"""
    model = GradientBoostingRegressor(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        n_estimators=5000, random_state=SEED,
        validation_fraction=0.2,
        n_iter_no_change=50, tol=1e-4
    )
    model.fit(X_train, y_train)
    return model

# ==================== 主循环 ====================
all_model_summaries = {}  # 用于汇总

for model_name, features in model_configs.items():
    print(f"\n{'='*50}")
    print(f"开始模型: {model_name}")
    model_dir = os.path.join(ROOT, model_name)
    os.makedirs(os.path.join(model_dir, "交叉验证"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "滚动预测"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "变量重要性"), exist_ok=True)

    # 用于收集全部预测以计算总体R²
    all_y_true = []
    all_y_pred = []
    all_importances = []  # 存储每年重要性DataFrame

    summary = []  # 该模型各轮测试结果

    for train_year, test_year in zip(TRAIN_YEARS, TEST_YEARS):
        # 准备数据
        X_train, y_train, _ = prepare_data(df, features, TARGET, train_year)
        X_test, y_test, test_meta = prepare_data(df, features, TARGET, test_year)

        if len(X_train) < 50 or len(X_test) < 10:
            print(f"警告: {model_name} {train_year}→{test_year} 数据量不足，跳过")
            continue

        # 交叉验证
        cv_df, best_params = perform_cv(X_train, y_train)
        cv_path = os.path.join(model_dir, "交叉验证", f"{train_year}_CV结果.csv")
        cv_df.to_csv(cv_path, index=False, encoding='utf-8-sig')

        # 训练最终模型
        final_model = train_final_model(X_train, y_train, best_params)

        # 变量重要性
        imp_df = pd.DataFrame({
            'feature': features,
            'importance': final_model.feature_importances_
        })
        imp_path = os.path.join(model_dir, "变量重要性", f"{train_year}_重要性.csv")
        imp_df.to_csv(imp_path, index=False, encoding='utf-8-sig')
        all_importances.append(imp_df)

        # 预测
        y_pred = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        summary.append((test_year, test_r2, test_rmse))

        # 保存预测
        pred_df = pd.DataFrame({
            'Stkcd': test_meta['Stkcd'],
            'Year': test_year,
            'y_true': y_test.values,
            'y_pred': y_pred
        })
        pred_path = os.path.join(model_dir, "滚动预测", f"{train_year}_预测.csv")
        pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

        # 收集用于总体R²
        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)

        print(f"  [{model_name}] {train_year}→{test_year} 测试 R² = {test_r2:.6f}")

    # 存储本模型汇总信息
    all_model_summaries[model_name] = {
        'summary': summary,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'importances': all_importances
    }

# ==================== 生成汇总报告 ====================
report_lines = []
report_lines.append("NPro 模型滚动窗口汇总")
report_lines.append("=" * 70)

for model_name in ['NPro_基准', 'NPro_CEO', 'NPro_董事长']:
    info = all_model_summaries[model_name]
    report_lines.append(f"\n【{model_name}】")
    report_lines.append(f"{'年份':<6} {'测试R²':<10} {'RMSE':<10}")
    report_lines.append("-" * 30)
    for test_year, r2, rmse in info['summary']:
        report_lines.append(f"{test_year:<6} {r2:<10.6f} {rmse:<10.6f}")

    # 总体R²
    y_true = np.array(info['y_true'])
    y_pred = np.array(info['y_pred'])
    if len(y_true) > 0:
        overall_r2 = r2_score(y_true, y_pred)
        report_lines.append(f"总体R²: {overall_r2:.6f}")
    else:
        report_lines.append("总体R²: 无有效测试数据")

    # 平均变量重要性（取前5）
    if info['importances']:
        all_imp_df = pd.concat(info['importances'])
        mean_imp = all_imp_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        total_imp = mean_imp.sum()
        mean_imp_pct = (mean_imp / total_imp * 100).round(2)
        report_lines.append("\n平均变量重要性（前5）:")
        for i, (feat, pct) in enumerate(mean_imp_pct.head(5).items()):
            report_lines.append(f"  {i+1}. {feat}: {pct}%")

report_path = os.path.join(ROOT, "NPro_汇总结果.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"\n汇总报告已保存至: {report_path}")

# ==================== 抽样文件生成 ====================
sample_dir = os.path.join(ROOT, "抽样")
os.makedirs(sample_dir, exist_ok=True)

for model_name in ['NPro_基准', 'NPro_CEO', 'NPro_董事长']:
    # 预测抽样：找到该模型所有预测文件，合并后抽取50条
    pred_dir = os.path.join(ROOT, model_name, "滚动预测")
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('_预测.csv')]
    pred_dfs = []
    for fname in pred_files:
        path = os.path.join(pred_dir, fname)
        pred_dfs.append(pd.read_csv(path))
    if pred_dfs:
        combined_pred = pd.concat(pred_dfs, ignore_index=True)
        sample_pred = combined_pred.sample(n=min(50, len(combined_pred)), random_state=SEED)
        sample_pred.to_csv(os.path.join(sample_dir, f"{model_name}_预测抽样.csv"), index=False, encoding='utf-8-sig')

    # 重要性抽样：取首、中、尾年份的重要性，合并后抽30条
    imp_dir = os.path.join(ROOT, model_name, "变量重要性")
    imp_files = sorted([f for f in os.listdir(imp_dir) if f.endswith('_重要性.csv')])
    if len(imp_files) >= 3:
        selected = [imp_files[0], imp_files[len(imp_files)//2], imp_files[-1]]
    elif len(imp_files) > 0:
        selected = imp_files
    else:
        continue
    imp_dfs = []
    for fname in selected:
        path = os.path.join(imp_dir, fname)
        imp_dfs.append(pd.read_csv(path))
    combined_imp = pd.concat(imp_dfs, ignore_index=True)
    sample_imp = combined_imp.sample(n=min(30, len(combined_imp)), random_state=SEED)
    sample_imp.to_csv(os.path.join(sample_dir, f"{model_name}_重要性抽样.csv"), index=False, encoding='utf-8-sig')

print(f"抽样文件已保存至: {sample_dir}")
print("全部任务完成。")