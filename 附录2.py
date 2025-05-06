import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

# 1. 数据加载与预处理
try:
    df = pd.read_csv('student-por.csv', sep=';')
except FileNotFoundError:
    print("❌ 错误：请将student-por.csv放在当前目录")
    print("📌 下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip")
    exit()

# 数据清洗与特征工程
df['G3_class'] = (df['G3'] >= 10).astype(int)
selected_features = ['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel']
df = df[selected_features + ['G3_class']].dropna()

# 分割数据集
X, y = df[selected_features], df['G3_class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20%作为测试集
    random_state=42,         # 固定随机种子
    stratify=y               # 保持类别分布
)

# 2. 训练最佳深度模型
best_depth = 4  # 已知最佳深度
model = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

# 3. 预测并展示结果
y_pred = model.predict(X_test)

# 选择前10个测试样本展示
test_samples = X_test.head(10).copy()
test_samples['预测结果'] = y_pred[:10]
test_samples['真实结果'] = y_test.head(10).values

print("\n==================== 测试样本结果展示 ====================")
print(test_samples[['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel', '预测结果', '真实结果']])
