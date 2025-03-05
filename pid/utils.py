import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def clustering(X, pca=False, n_components=5, n_clusters=20):
  X = np.nan_to_num(X)
  if pca:
    X = normalize(X)
    X = PCA(n_components=n_components).fit_transform(X)
  kmeans = KMeans(n_clusters=n_clusters).fit(X)
  return kmeans.labels_, X

