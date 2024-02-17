import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)
labels = ms.labels_
clustercenters = ms.cluster_centers_
nclusters = len(clustercenters)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.title('Исходные точки')
plt.subplot(122)
plt.scatter(clustercenters[:, 0], clustercenters[:, 1], color='r')
plt.title('Центры кластеров')

scores = []
for n in range(2, 16):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_)
    scores.append(score)

plt.figure(figsize=(6, 6))
plt.bar(range(2, 16), scores)
plt.xlabel('Количество кластеров')
plt.ylabel('Score')
plt.title('Бар диаграмма score(number of clusters)')

optimal_n_clusters = scores.index(max(scores)) + 2
kmeans = KMeans(n_clusters=optimal_n_clusters)
kmeans.fit(X)

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='r')
plt.title('Кластеризованные данные с областями кластеризации')

plt.show()
