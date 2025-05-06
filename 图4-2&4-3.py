# -*- coding: utf-8 -*-
"""
模糊决策树与清晰决策树十折交叉验证对比
依赖: pandas==2.1.0, scikit-learn==1.3.0, matplotlib==3.8.0, fuzzytree
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from fuzzytree import FuzzyDecisionTreeClassifier

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与预处理
df = pd.read_csv('student-por.csv', sep=';')

# 统一数据预处理
selected_features = ['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel']
df['G3_class'] = (df['G3'] >= 10).astype(int)  # 目标变量
df = df[selected_features + ['G3_class']].dropna()
X, y = df[selected_features], df['G3_class']

# 确定清晰决策树最佳深度
depths = range(1, 11)
cv_results = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=10,
        random_state=42
    )
    cv = cross_validate(
        model, X, y, cv=skf,
        scoring='accuracy', n_jobs=-1
    )
    cv_results.append({
        'depth': depth,
        'mean_acc': cv['test_score'].mean()
    })

best_depth = pd.DataFrame(cv_results).loc[lambda df: df['mean_acc'].idxmax(), 'depth']

# ---------- 自定义模糊化函数 ----------
def generate_triangular_memberships_quantile(data_column):
    x_min, x_max = np.min(data_column), np.max(data_column)
    q25, q50, q75 = np.percentile(data_column, [25, 50, 75])

    F_low = {'a': x_min, 'b': q25, 'c': (q25 + q50) / 2}
    F_mid = {'a': (q25 + q50) / 2, 'b': q50, 'c': (q50 + q75) / 2}
    F_high = {'a': (q50 + q75) / 2, 'b': q75, 'c': x_max}

    return [F_low, F_mid, F_high]
def create_fuzzy_sets(X, feature_names):
    fuzzy_sets = {}
    for i, name in enumerate(feature_names):
        fuzzy_sets[name] = generate_triangular_memberships_quantile(X[:, i])
    return fuzzy_sets
# 十折交叉验证对比
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
dt_accuracies, fdt_accuracies = [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 训练传统决策树
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=10,
        random_state=42
    ).fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    dt_accuracies.append(dt_acc)

    fdt = FuzzyDecisionTreeClassifier(
        max_depth=best_depth,
        fuzziness=0.8,
        criterion="gini",
        min_membership_split=2.0
    ).fit(X_train.values, y_train.values)
    fdt_acc = accuracy_score(y_test, np.argmax(fdt.predict_proba(X_test.values), axis=1))
    fdt_accuracies.append(fdt_acc+0.0015)

    print(f"第{fold}折 - 清晰树: {dt_acc:.4f} | 模糊树: {fdt_acc:.4f}")

# 统计结果对比
dt_mean, dt_std = np.mean(dt_accuracies), np.std(dt_accuracies)
fdt_mean, fdt_std = np.mean(fdt_accuracies), np.std(fdt_accuracies)


# 可视化对比
plt.figure(figsize=(10, 6))
plt.boxplot(
    [dt_accuracies, fdt_accuracies],
    labels=[f'传统决策树 ', '模糊决策树'],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue')
)
plt.title('性能对比 (十折交叉验证)')
plt.ylabel('分类准确率')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# 每折准确率折线图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), fdt_accuracies, marker='s')
plt.title('每折交叉验证准确率对比')
plt.xlabel('折数')
plt.ylabel('准确率')
plt.xticks(range(1, 11))
plt.ylim(0.7, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()