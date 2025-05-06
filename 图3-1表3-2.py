# -*- coding: utf-8 -*-
"""
十折交叉验证剪枝效果分析（修复版）
解决cross_val_score不支持return_train_score的问题
依赖: pandas==2.1.0, scikit-learn==1.3.0, matplotlib==3.8.0
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理（含错误处理）
try:
    df = pd.read_csv('student-por.csv', sep=';')
except FileNotFoundError:
    print("❌ 错误：请将student-por.csv放在当前目录")
    print("📌 下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip")
    exit()

# 数据清洗与特征工程
df['G3_class'] = (df['G3'] >= 10).astype(int)  # 及格线10分
selected_features = ['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel']  # TOP6特征
df = df[selected_features + ['G3_class']].dropna()  # 移除缺失值
if len(df) < 10:  # 确保十折验证有足够样本
    print("❌ 错误：清洗后样本数不足10，无法进行十折验证")
    exit()

X, y = df[selected_features], df['G3_class']

# 2. 十折分层交叉验证（修复版核心）
depths = range(1, 11)  # 测试深度1-10
cv_results = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # 分层抽样保持类别分布

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=10,
        random_state=42
    )

    # 使用cross_validate替代cross_val_score（修复TypeError）
    cv = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=skf,
        scoring=make_scorer(accuracy_score),
        n_jobs=-1,  # 使用所有CPU核心加速
        return_train_score=False  # 仅需要测试集分数
    )

    # 训练全量数据获取叶节点数（代表树复杂度）
    full_model = model.fit(X, y)
    cv_results.append({
        'depth': depth,
        'leaves': full_model.get_n_leaves(),  # 叶节点数
        'mean_acc': cv['test_score'].mean(),  # 十折平均准确率
        'std_acc': cv['test_score'].std(),  # 准确率标准差
        'min_acc': cv['test_score'].min(),  # 最低单折准确率
        'max_acc': cv['test_score'].max()  # 最高单折准确率
    })

# 3. 结果分析
cv_df = pd.DataFrame(cv_results)
best_idx = cv_df['mean_acc'].idxmax()  # 最佳深度索引
best_depth = cv_df.loc[best_idx, 'depth']

# 4. 可视化：深度-准确率关系（含误差条+叶节点标注）
plt.figure(figsize=(19, 8))

# 主图：准确率折线（带误差条）
plt.errorbar(
    x=cv_df['depth'],
    y=cv_df['mean_acc'],
    yerr=cv_df['std_acc'],
    fmt='o-',
    color='#2E86C1',
    ecolor='#3498DB',
    elinewidth=2,
    capsize=8,
    label=f'十折平均准确率（±标准差）'
)

# 次轴：叶节点数（右侧Y轴）
ax2 = plt.twinx()
ax2.plot(
    cv_df['depth'],
    cv_df['leaves'],
    's--',
    color='#E74C3C',
    label='叶节点数量'
)
ax2.set_ylabel('叶节点数量', color='#E74C3C', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#E74C3C')

# 标注最佳深度
best_row = cv_df.iloc[best_idx]
plt.scatter(
    best_depth, best_row['mean_acc'],
    s=200, color='green', marker='*',
    edgecolors='white', linewidth=2,
)

# 图表美化
plt.title("剪枝深度对准确率的影响（十折交叉验证）", fontsize=16, pad=25)
plt.xlabel("决策树深度（剪枝程度）", fontsize=13)
plt.ylabel("平均准确率", color='#2E86C1', fontsize=13)
plt.xticks(depths, labels=[f'D{i}' for i in depths])
plt.grid(True, linestyle='--', alpha=0.6, which='both')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=11)
plt.show()