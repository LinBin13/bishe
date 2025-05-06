import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from fuzzytree import FuzzyDecisionTreeClassifier
from deap import base, creator, tools, algorithms
import random

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("student-por.csv", sep=';',
                 usecols=['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel', 'G3'])
df = df.dropna().reset_index(drop=True)
df['G3_bin'] = (df['G3'] >= 10).astype(int)

X = df[['failures', 'studytime', 'absences', 'age', 'Walc', 'famrel']].values
y = df['G3_bin'].values

feature_num = X.shape[1]


# === 分位数方法构建模糊函数参数 ===
def quantile_fuzzify_params(X):
    params = []
    for col in X.T:
        q25, q50, q75 = np.percentile(col, [25, 50, 75])
        xmin, xmax = np.min(col), np.max(col)
        # 3 个模糊集，每个是 a, b, c
        F_low = (xmin, q25, (q25 + q50) / 2)
        F_mid = ((q25 + q50) / 2, q50, (q50 + q75) / 2)
        F_high = ((q50 + q75) / 2, q75, xmax)
        params.extend([F_low, F_mid, F_high])
    return np.array(params).reshape(feature_num, 3, 3)


# === 模糊化数据 ===
def fuzzify_data(X, fuzzy_params):
    n_samples, n_features = X.shape
    result = []
    for i in range(n_features):
        params = fuzzy_params[i]
        col = X[:, i]
        col_fuzz = []
        for (a, b, c) in params:
            μ = np.where(
                col <= a, 0,
                np.where(col <= b, (col - a) / (b - a + 1e-6),
                         np.where(col <= c, (c - col) / (c - b + 1e-6), 0))
            )
            col_fuzz.append(μ)
        result.extend(col_fuzz)
    return np.vstack(result).T  # shape (n_samples, n_features * 3)


# === 方法1：分位数方法 ===
start_time = time.time()
quantile_params = quantile_fuzzify_params(X)
X_quantile_fuzzified = fuzzify_data(X, quantile_params)

X_train, X_test, y_train, y_test = train_test_split(X_quantile_fuzzified, y, test_size=0.3, random_state=42)
model = FuzzyDecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_quantile = accuracy_score(y_test, y_pred)
time_quantile = time.time() - start_time


# === 方法2：遗传算法 ===
# 决策树训练函数
def evaluate_individual(individual):
    params = np.array(individual).reshape(feature_num, 3, 3)
    X_fuzz = fuzzify_data(X, params)
    scores = cross_val_score(FuzzyDecisionTreeClassifier(max_depth=3), X_fuzz, y, cv=5)
    return scores.mean(),


# 设置遗传算法参数
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# 生成初始个体（按特征 min-max 范围）
param_bounds = []
for col in X.T:
    xmin, xmax = np.min(col), np.max(col)
    for _ in range(3):  # 每个模糊集
        param_bounds.extend([(xmin, xmax)] * 3)


def uniform_param(low, high):
    return random.uniform(low, high)


toolbox.register("attr_float", lambda: uniform_param(*random.choice(param_bounds)))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=feature_num * 3 * 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(42)

start_time = time.time()

pop = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                               stats=stats, halloffame=hof, verbose=False)

best_params = np.array(hof[0]).reshape(feature_num, 3, 3)
X_genetic_fuzzified = fuzzify_data(X, best_params)

X_train, X_test, y_train, y_test = train_test_split(X_genetic_fuzzified, y, test_size=0.3, random_state=42)
model = FuzzyDecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_genetic = accuracy_score(y_test, y_pred)
time_genetic = time.time() - start_time

# === 输出对比结果 ===
print(f"✅ 分位数方法准确率: {acc_quantile:.4f}，耗时: {time_quantile:.2f} 秒")
print(f"🧬 遗传算法方法准确率: {acc_genetic:.4f}，耗时: {time_genetic:.2f} 秒")

# === 准确率 & 时间 对比图 ===
labels = ['分位数方法', '遗传算法']
accuracies = [acc_quantile, acc_genetic]
times = [time_quantile, time_genetic]

fig, ax1 = plt.subplots(figsize=(8, 5))

bar_width = 0.35
x = np.arange(len(labels))

# 左轴：准确率
ax1.bar(x - bar_width / 2, accuracies, bar_width, label='准确率', color='skyblue')
ax1.set_ylabel('准确率')
ax1.set_ylim(0, 1)

# 右轴：耗时
ax2 = ax1.twinx()
ax2.bar(x + bar_width / 2, times, bar_width, label='耗时 (秒)', color='salmon')
ax2.set_ylabel('耗时 (秒)')

# X轴标签
ax1.set_xticks(x)
ax1.set_xticklabels(labels)

# 图例
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

plt.title("分位数方法 vs 遗传算法：准确率 & 耗时对比")
plt.tight_layout()
plt.show()

# === 绘制遗传算法每一代的平均准确率折线图 ===
gen = log.select("gen")
avg_accuracies = log.select("avg")

plt.figure(figsize=(8, 5))
plt.plot(gen, avg_accuracies, marker='o', linestyle='-', color='blue')
plt.title("遗传算法每一代的平均准确率")
plt.xlabel("迭代次数")
plt.ylabel("平均准确率")
plt.grid(False)
plt.show()