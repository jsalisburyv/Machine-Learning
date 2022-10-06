import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
# TODO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Estandaritzar les dades: StandardScaler
# TODO
scaler = StandardScaler()
X_train_trans = scaler.fit_transform(X_train)
X_test_trans = scaler.transform(X_test)

# Entrenam una SVM linear (classe SVC)
# TODO
svc = SVC(kernel="linear", C=1000, random_state=42)
svc.fit(X_train_trans, y_train)


# Prediccio
# TODO
y_pred = svc.predict(X_test_trans)

# Metrica
# TODO
accuracy = np.count_nonzero(y_pred == y_test)/len(y_test)
print(accuracy)