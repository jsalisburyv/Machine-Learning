
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os


# ### Read Preprocessed Data

# for every file in data/comb open the csv
# and append it to a list
scaler = StandardScaler()  # try different scalers
pca = PCA(n_components=0.99, random_state=43)  # try different n_components
# Create common Models with different Kernels
models = [
    svm.SVC(kernel='linear', probability=True),  # linear kernel
    # polynomial kernel with degree 3
    svm.SVC(kernel='poly', probability=True),
    svm.SVC(kernel='rbf', probability=True),  # RBF kernel with gamma = 0.1
    svm.SVC(kernel='sigmoid', probability=True)
]

for file in os.listdir('../data/comb'):
    print(file.split('.')[0])
    df = pd.read_csv('../data/comb/'+file, index_col=0)
    # Separate features from target
    X = df.drop(['word', 'lang'], axis=1)
    y = df['lang']
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, random_state=43)

    # Scale Data: https://towardsdatascience.com/feature-scaling-and-normalisation-in-a-nutshell-5319af86f89b
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Dimensionality Reduction: https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # Train models
    for model in models:
        print("Fitting model: ", model)
        model.fit(X_train_pca, y_train)

    for i, model in enumerate(models):
        prediction = model.predict_proba(X_test_pca)
        print(
            f'Top 3 accuracy model {i+1}: {round(top_k_accuracy_score(y_test, prediction, k=3)*100,2)}%')
        # print accuracy, precision, f1-score
        langs = ['en', 'de', 'ca', 'es', 'fr', 'it', 'pl', 'pt', 'ru', 'sv']
        report = classification_report(y_test, model.predict(
            X_test_pca), target_names=langs, output_dict=True)
        print("Accuracy: ", round(report['accuracy']*100, 2), "Precision: ", round(
            report['macro avg']['precision']*100, 2), "F1-score: ", round(report['macro avg']['f1-score']*100, 2))
