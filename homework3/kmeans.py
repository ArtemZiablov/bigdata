import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import pairwise_distances_argmin_min
import time
import os


# Кількість ядер процесора
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 1. Створення даних для опуклих кластерів
X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 2. Створення даних для неопуклих кластерів
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=0)

# Функція для візуалізації результатів кластеризації
def visualize_clusters(X, y_kmeans, centers, title):
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.title(title)
    plt.show()

# Функція для аналізу зміни відстаней всередині кластерів
def analyze_inertia(X, n_clusters, max_iter=300, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state, n_init=10)
    kmeans.fit(X)
    distances = []
    for i in range(max_iter):
        _, min_distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        distances.append(np.mean(min_distances))
        kmeans.max_iter = i + 1
        kmeans.fit(X)
    return distances

# Приклад 1: Опуклі кластери (працює добре)
kmeans_blobs = KMeans(n_clusters=4, random_state=42)
start_time = time.time()
y_kmeans_blobs = kmeans_blobs.fit_predict(X_blobs)
execution_time_blobs = time.time() - start_time
visualize_clusters(X_blobs, y_kmeans_blobs, kmeans_blobs.cluster_centers_, "Опуклі кластери (працює добре)")

# Приклад 2: Неопуклі кластери (працює погано)
kmeans_moons = KMeans(n_clusters=2, random_state=42)
start_time = time.time()
y_kmeans_moons = kmeans_moons.fit_predict(X_moons)
execution_time_moons = time.time() - start_time
visualize_clusters(X_moons, y_kmeans_moons, kmeans_moons.cluster_centers_, "Неопуклі кластери (працює погано)")

# Вплив параметрів
params = [5, 10, 20]
for n_init in params:
    kmeans = KMeans(n_clusters=4, n_init=n_init, random_state=42)
    start_time = time.time()
    kmeans.fit(X_blobs)
    execution_time = time.time() - start_time
    print(f"n_init={n_init}, час виконання: {execution_time:.4f} секунд")

# Аналіз зміни відстаней всередині кластерів
distances_blobs = analyze_inertia(X_blobs, n_clusters=4)
plt.plot(range(1, len(distances_blobs) + 1), distances_blobs)
plt.title("Зміна середньої відстані всередині кластерів (опуклі)")
plt.xlabel("Ітерації")
plt.ylabel("Середня відстань")
plt.show()

distances_moons = analyze_inertia(X_moons, n_clusters=2)
plt.plot(range(1, len(distances_moons) + 1), distances_moons)
plt.title("Зміна середньої відстані всередині кластерів (неопуклі)")
plt.xlabel("Ітерації")
plt.ylabel("Середня відстань")
plt.show()
