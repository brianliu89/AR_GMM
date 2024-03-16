from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import math

features = np.load('features.npy')

n_components = 3

pca = PCA(n_components=n_components)
features_2d = pca.fit_transform(features)

gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=0)
gmm.fit(features_2d)

labels = gmm.predict(features_2d)

means = gmm.means_
covariances = gmm.covariances_

color = ["#FF0000", "#FF0088", "#00FF00", "#00FFFF", "#0000FF",
         "#FF00FF", "#FFFF00", "#FF8800", "#00FF99", "#7700FF"]

plt.rcParams["figure.figsize"] = (18,18)
fig, ax = plt.subplots()

n_components = len(means)  # 獲取GMM模型中分量的數量

# spherical
for i in range(n_components):
    bool_idx = (labels == i)
    ax.scatter(features_2d[bool_idx, 0], features_2d[bool_idx, 1], c=color[i], label=str(i))
    variance = covariances[i]
    for j in [2, 3]:
        radius = j * math.sqrt(variance)
        circle = plt.Circle(xy=means[i], radius=radius, color=color[i], alpha=0.2/j)
        ax.add_artist(circle)

# diag
# for i in [2,3,6]:  # 使用组件数量来生成索引
#     bool_idx = (labels == i)
#     print(f"Component {i}, Points: {bool_idx.sum()}")
#     if bool_idx.sum() == 0:
#         continue
#     ax.scatter(features_2d[bool_idx, 0], features_2d[bool_idx, 1], c=color[i % len(color)], label=str(i))
#     variances = covariances[i]
#     print('means: ', means[i])
#     print('covariances: ', covariances[i])
#     for j in [2, 3]:
#         width = j * math.sqrt(variances[0])
#         height = j * math.sqrt(variances[1])
#         angle = 0
#         ellipse = Ellipse(xy=means[i], width=width, height=height, angle=angle, color=color[i], alpha=0.2/j)
#         ax.add_artist(ellipse)

plt.xticks([])
plt.yticks([])
plt.legend(fontsize=10)
plt.savefig('gmm_density_contours.png')
plt.show()
