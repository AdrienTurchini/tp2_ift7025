import numpy as np
import sys

from load_datasets import load_iris_dataset
from load_datasets import load_wine_dataset
from load_datasets import load_abalone_dataset

from NaiveBayes import NaiveBayes
from Knn import Knn

from labelEncoder import labelEncoder
import operator

def cv_knn(N_split, X_train, y_train, X_test, y_test, K_values):

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Splits
    X_splits = np.split(X, N_split)
    y_splits = np.split(y, N_split)

    # Recherche
    scores = {}
    for K in K_values:
        accuracys = []
        for i in range(N_split):
            X_train = np.concatenate(np.delete(X_splits, i, 0))
            X_test = X_splits[i]
            y_train = np.concatenate(np.delete(y_splits, i, 0))
            y_test = y_splits[i]

            model = Knn(K=K)
            model.train(X_train, y_train)
            evaluate = model.evaluate(X_test, y_test)
            
            accuracys.append(evaluate['mean_accuracy'])

        scores[K] = np.mean(accuracys)

    # print + selection
    print(max(scores.items(), key=operator.itemgetter(1)))
    return max(scores.items(), key=operator.itemgetter(1))[0]

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initialisez vos paramètres
K_values = [k for k in range(1, 21)]

# Initialisez/instanciez vos classifieurs avec leurs paramètres


# Charger/lire les datasets
X_train_iris, y_train_iris, X_test_iris, y_test_iris = load_iris_dataset(train_ratio=0.7, seed=42, shuffle=True)
X_train_wine, y_train_wine, X_test_wine, y_test_wine = load_wine_dataset(train_ratio=0.7, seed=42, shuffle=True)
X_train_abalone, y_train_abalone, X_test_abalone, y_test_abalone = load_abalone_dataset(train_ratio=0.7, seed=42, shuffle=True)

# Encode Iris y
le = labelEncoder()
y_train_iris = le.fit_transform(y_train_iris)
y_test_iris = le.transform(y_test_iris)

# Validation croisée
N_split_iris = 6 # --> Pas toutes les classes das les splits si > 6
N_split_wine = 10
N_split_abalone = 10

print('IRIS')
K_opti_iris = cv_knn(N_split_iris, X_train_iris, y_train_iris, X_test_iris, y_test_iris, K_values)
print('WIN')
K_opti_wine = cv_knn(N_split_wine, X_train_wine, y_train_wine, X_test_wine, y_test_wine, K_values)
print('ABALONE')
K_opti_abalone = cv_knn(N_split_abalone, X_train_abalone, y_train_abalone, X_test_abalone, y_test_abalone, K_values)

# Entrainez votre classifieur
clf_Knn = Knn(K=K_opti_iris)
clf_Knn.train(X_train_iris, y_train_iris)

"""
Après avoir fait l'entrainement, évaluez votre modèle sur 
les données d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
# Tester votre classifieur
print('---------- IRIS TRAIN ----------')
evaluate_train = clf_Knn.evaluate(X_train_iris, y_train_iris)
for e in evaluate_train:
    print(f'{e}\n {evaluate_train[e]}')

print('---------- WIN TRAIN ----------')
evaluate_train = clf_Knn.evaluate(X_train_iris, y_train_iris)
for e in evaluate_train:
    print(f'{e}\n {evaluate_train[e]}')

print('---------- ABALONE TRAIN ----------')
evaluate_train = clf_Knn.evaluate(X_train_iris, y_train_iris)
for e in evaluate_train:
    print(f'{e}\n {evaluate_train[e]}')
"""
Finalement, évaluez votre modèle sur les données de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la précision (precision)
    - le rappel (recall)
    - le F1-score
"""
print('---------- IRIS TEST ----------')
evaluate_test = clf_Knn.evaluate(X_test_iris, y_test_iris)
for e in evaluate_test:
    print(f'{e}\n {evaluate_test[e]}')

print('---------- WIN TEST ----------')
evaluate_test = clf_Knn.evaluate(X_test_iris, y_test_iris)
for e in evaluate_test:
    print(f'{e}\n {evaluate_test[e]}')

print('---------- ABALONE TEST ----------')
evaluate_test = clf_Knn.evaluate(X_test_iris, y_test_iris)
for e in evaluate_test:
    print(f'{e}\n {evaluate_test[e]}')


