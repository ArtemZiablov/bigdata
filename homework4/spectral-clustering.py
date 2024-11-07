import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_circles

# Генерація даних у вигляді концентричних кіл
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=13)

# Виконання спектральної кластеризації
spectral_clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', random_state=13)
spectral_labels = spectral_clustering.fit_predict(X)

# Виконання кластеризації методом K-середніх
kmeans = KMeans(n_clusters=2, random_state=13)
kmeans_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Візуалізація результатів спектральної кластеризації
plt.figure(figsize=(14, 7))

# Спектральна кластеризація
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
plt.title('Spectral Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Кластеризація методом K-середніх
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='plasma', marker='o', edgecolor='k', alpha=0.7)
plt.title('K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
