import numpy as np
import pandas as pd
from fuzzytree import FuzzyDecisionTreeClassifier  # 经过修改的修改版
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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

# ---------- 数据读取 ----------
file_path = "student-por.csv"
df = pd.read_csv(file_path, sep=';', usecols=['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel', 'G3'])
df = df.dropna().reset_index(drop=True)
df['G3_bin'] = (df['G3'] >= 10).astype(int)

X = df[['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel']].values
y = df['G3_bin'].values
feature_names = ['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel']

# ---------- 创建模糊集 ----------
fuzzy_sets_all = create_fuzzy_sets(X, feature_names)

# ---------- 交叉验证 ----------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
membership = []
special_samples = []
failures_values = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = FuzzyDecisionTreeClassifier(
        fuzziness=0.8,
        criterion="gini",
        max_depth=5,
        min_membership_split=2.0,
        fuzzy_sets=fuzzy_sets_all,  # 新增参数：自定义模糊集
        feature_names=feature_names
    )
    clf.fit(X_train, y_train)
    y_pred_prob = clf.predict_proba(X_test)

    accuracies.append(accuracy_score(y_test, np.argmax(y_pred_prob, axis=1)))
    membership.append(y_pred_prob[:5])

    print(f"\n第{fold}折:")
    header = " ".join([f"{name:<10}" for name in feature_names]) + "隶属度向量"
    print(header)

    for i in range(5):
        if i >= len(X_test):
            break
        feature_values = [f"{val:<10.2f}" for val in X_test[i]]
        membership_str = np.array2string(y_pred_prob[i], precision=4, separator=', ')
        print(" ".join(feature_values) + membership_str)

    if fold == 3 and len(X_test) > 0:
        special_samples.append((X_test[0], y_pred_prob[0]))
        failures_values.append(X_test[0][0])
    if fold == 4 and len(X_test) > 2:
        special_samples.append((X_test[2], y_pred_prob[2]))
        failures_values.append(X_test[2][0])
    if fold in [1, 5, 6]:
        for i in range(len(X_test)):
            if X_test[i][0] not in failures_values:
                special_samples.append((X_test[i], y_pred_prob[i]))
                failures_values.append(X_test[i][0])
                if len(special_samples) >= 5:
                    break
    if len(special_samples) >= 5:
        break

# ---------- 输出统计 ----------
print("\n每折准确率：", np.round(accuracies, 4))
print(f"\n平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print("\n===== 选择样本分析 =====")
header = " ".join([f"{name:<10}" for name in feature_names]) + "隶属度向量"
print(header)

for feat, memb in special_samples[:5]:
    feature_values = [f"{val:<10.2f}" for val in feat]
    membership_str = np.array2string(memb, precision=4, separator=', ')
    print(" ".join(feature_values) + membership_str)
