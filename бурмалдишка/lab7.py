import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Генерация данных
X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)

# График 1: исходные кластеры
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="white", linewidth=0.8, s=60)
plt.title("Исходные кластеры данных", fontsize=14)
plt.xlabel("Признак 1", fontsize=12)
plt.ylabel("Признак 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# Инициализация параметров модели
sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2)
b = 0.0

def forward(X):
    return sig(X @ w + b)

print("Точность до обучения:", ((forward(X) > 0.5) == y).mean())

def bce(p, y):
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

print("Исходная ошибка (Loss):", bce(forward(X), y))

# Цикл обучения (Градиентный спуск)
lr, hist = 0.1, []
for i in range(500):
    p = forward(X)
    dz = p - y
    w -= lr * X.T @ dz / len(X)
    b -= lr * dz.mean()
    hist.append(bce(p, y))

# лфункция потерь
plt.figure(figsize=(7, 5))
plt.plot(hist, color="#2E86AB", linewidth=2.2)
plt.title("График функции потерь (BCE Loss)", fontsize=14)
plt.xlabel("Эпохи обучения", fontsize=12)
plt.ylabel("Ошибка (Loss)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()

print("Точность после обучения:", ((forward(X) > 0.5) == y).mean())

# Подготовка сетки для разделяющей границы
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)  # испра
    )
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# разделяющая граница
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.4)
plt.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=2.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="white", linewidth=1, s=70)
plt.title("Разделяющая граница классов", fontsize=14)
plt.xlabel("Признак 1", fontsize=12)
plt.ylabel("Признак 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.2)
plt.tight_layout()
plt.show()
