import numpy as np
from scipy.spatial import distance_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

def kernel_lineal(x1, x2):
    return x1.dot(x2.T)

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=42)
y[y == 0] = -1

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)


# Feina1
# TODO
X_xor_transformed =  scaler.fit_transform(X)
svc_def = SVC(kernel="linear", C=1, random_state=42)
svc_def.fit(X_xor_transformed, y)
y_pred_def = svc_def.predict(X_xor_transformed)

accuracy = np.count_nonzero(y_pred_def == y)/len(y)
print(f"default linear = {accuracy}")

svc_cus = SVC(kernel=kernel_lineal, C=1, random_state=42)
svc_cus.fit(X_xor_transformed, y)
y_pred_cus = svc_cus.predict(X_xor_transformed)

accuracy = np.count_nonzero(y_pred_cus == y)/len(y)
print(f"custom linear = {accuracy}\n")

# Feina2
# TODO
def kernel_gaussia(x1, x2, gamma=1):
    return np.exp(-gamma * distance_matrix(x1, x2)**2)

np.random.seed(33)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

scaler = StandardScaler()
X_xor_transformed = scaler.fit_transform(X_xor)

svc_def = SVC(kernel='rbf', C=1, gamma=1,  random_state=42)
svc_def.fit(X_xor_transformed, y_xor)
y_pred_def = svc_def.predict(X_xor_transformed)

accuracy = np.count_nonzero(y_pred_def == y_xor)/len(y_xor)
print(f"default gaussian = {accuracy}")


svc_cus = SVC(kernel=kernel_gaussia, C=1, gamma=1,  random_state=42)
svc_cus.fit(X_xor_transformed, y_xor)
y_pred_cus = svc_cus.predict(X_xor_transformed)

accuracy = np.count_nonzero(y_pred_cus == y_xor)/len(y_xor)
print(f"custom gaussian = {accuracy}\n")

# Feina3
# TODO
def kernel_polinomic(x1, x2, r=0, degree=3):
    return (x1.dot(x2.T) + r)**degree

svc_def = SVC(kernel='poly', C=1, gamma=1, coef0=0, degree=3,  random_state=42)
svc_def.fit(X_transformed, y)
y_pred_def = svc_def.predict(X_transformed)

accuracy = np.count_nonzero(y_pred_def == y)/len(y)
print(f"default polinomial = {accuracy}")


svc_cus = SVC(kernel=kernel_polinomic, C=1, gamma=1, coef0=0, degree=3,  random_state=42)
svc_cus.fit(X_xor_transformed, y_xor)
y_pred_cus = svc_cus.predict(X_xor_transformed)

accuracy = np.count_nonzero(y_pred_cus == y_xor)/len(y_xor)
print(f"custom polinomial = {accuracy}\n")

# Feina4
# TODO
polinomial = PolynomialFeatures(degree=3)
X_poly = polinomial.fit_transform(X_transformed)

svc_def = SVC(kernel='linear', C=1, gamma=1, coef0=0, degree=3,  random_state=42)
svc_def.fit(X_poly, y)
y_pred_def = svc_def.predict(X_poly)

accuracy = np.count_nonzero(y_pred_def == y)/len(y)
print(f"linear polinomial = {accuracy}")