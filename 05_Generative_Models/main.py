import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

os.environ['OMP_NUM_THREADS'] = '8' # per evitar warnings

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


def plot_gallery(images, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())

# 1. Carrega de dades
digits = datasets.load_digits()
scaler = StandardScaler()
plot_digits(digits.data[:100, :])
plt.show()
digits.data = scaler.fit_transform(digits.data)

# 2. Reduccio de la dimensionalitat
pca = PCA(random_state=42)
pca.fit(digits.data)
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cum_var)
plt.title("Explained Variance in PCA by Components")

def find_component_count(variances: list, min: float, change: float) -> int:
    for i, var in enumerate(variances):
        if var > min and (var - variances[i+1]) < change:
            return i+1
    return len(variances)

n_comps = find_component_count(cum_var, 0.95, 0.01)
plt.axvline(x = n_comps, color = 'r', label = 'min')
plt.show()

pca = PCA(n_components=n_comps, random_state=42)
data_pca = pca.fit_transform(digits.data)

# 3. Selecció del model: parametrització.
bics = list()
for i in range(1, 10):
    gmm = GaussianMixture(n_components=i, random_state=42)
    gmm.fit(data_pca)
    bics.append(gmm.bic(data_pca))
plt.plot(range(1,10), bics)
plt.title('BIC metric by component in GMM')
plt.show()

n_comps = find_component_count(bics, 0, 0.001)
gmm = GaussianMixture(n_components=n_comps, random_state=42)
gmm.fit(data_pca)

# 4. Generació de nous exemples.
X, y = gmm.sample(n_samples=100)
res = pca.inverse_transform(X)
res = scaler.inverse_transform(res)
plot_digits(res)
plt.show()


# EXTRA: Cares
faces = datasets.fetch_lfw_people()
scaler = StandardScaler()
plot_gallery(faces.images[:12, :])
plt.show()
faces.data = scaler.fit_transform(faces.data)

pca = PCA(random_state=42)
pca.fit(faces.data)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_comps = find_component_count(cum_var, 0.95, 0.01)
pca = PCA(n_components=n_comps, random_state=42)
data_pca = pca.fit_transform(faces.data)

bics = list()
for i in range(1, 10):
    gmm = GaussianMixture(n_components=i, random_state=42)
    gmm.fit(data_pca)
    bics.append(gmm.bic(data_pca))
n_comps = find_component_count(bics, 0, 0.001)
gmm = GaussianMixture(n_components=1, random_state=42)
gmm.fit(data_pca)

X, y = gmm.sample(n_samples=12)
res = pca.inverse_transform(X)
res = scaler.inverse_transform(res)
images = res.reshape((12, 62, 47))
plot_gallery(images)
plt.show()


