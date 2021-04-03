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

def cv_NaivesBayes_init_train(X_train, y_train):
    X_train = np.matrix(X_train)
    clf = NaiveBayes()
    clf.train(X_train, y_train)
    return clf
    

def cv_NaivesBayes_evaluate(clf, X_test, y_test):
    evaluation = clf.evaluate(X_test, y_test)
    return evaluation


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

###############################################################################################
#                                             KNN                                             #
###############################################################################################
# Initialisez vos paramètres
K_values_iris = [k for k in range(1, 11)]
K_values_wine = [k for k in range(1, 51)]
K_values_abalone = [k for k in range(1, 51)]

# Initialisez/instanciez vos classifieurs avec leurs paramètres


# Charger/lire les datasets
X_train_iris, y_train_iris, X_test_iris, y_test_iris = load_iris_dataset(train_ratio=0.7, seed=69, shuffle=True)
X_train_wine, y_train_wine, X_test_wine, y_test_wine = load_wine_dataset(train_ratio=0.7, seed=42, shuffle=False)
X_train_abalone, y_train_abalone, X_test_abalone, y_test_abalone = load_abalone_dataset(train_ratio=0.7, seed=42, shuffle=False)

# Pop une valeur de train car len(X_abalone) == 4177 est premier donc on ne peut pas split ...
X_train_abalone = np.delete(X_train_abalone, 0, axis=0)
y_train_abalone = np.delete(y_train_abalone, 0, axis=0)

# Encode Iris y
le = labelEncoder()
y_train_iris = le.fit_transform(y_train_iris)
y_test_iris = le.transform(y_test_iris)

# Validation croisée
N_split_iris = 10
N_split_wine = 3 # --> Trop long
N_split_abalone = 3 # --> Trop long

print('IRIS')
K_opti_iris = cv_knn(N_split_iris, X_train_iris, y_train_iris, X_test_iris, y_test_iris, K_values_iris)
print('WINE')
K_opti_wine = cv_knn(N_split_wine, X_train_wine, y_train_wine, X_test_wine, y_test_wine, K_values_wine)
print('ABALONE')
K_opti_abalone = cv_knn(N_split_abalone, X_train_abalone, y_train_abalone, X_test_abalone, y_test_abalone, K_values_abalone)

# Entrainez votre classifieur
clf_Knn_iris = Knn(K=K_opti_iris)
clf_Knn_iris.train(X_train_iris, y_train_iris)

clf_Knn_wine = Knn(K=K_opti_wine)
clf_Knn_wine.train(X_train_wine, y_train_wine)

clf_Knn_abalone = Knn(K=K_opti_abalone)
clf_Knn_abalone.train(X_train_abalone, y_train_abalone)

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
evaluate_train = clf_Knn_iris.evaluate(X_train_iris, y_train_iris)
for e in evaluate_train:
    print(f'{e}\n {evaluate_train[e]}')

print('---------- WINE TRAIN ----------')
evaluate_train = clf_Knn_wine.evaluate(X_train_wine, y_train_wine)
for e in evaluate_train:
    print(f'{e}\n {evaluate_train[e]}')

print('---------- ABALONE TRAIN ----------')
evaluate_train = clf_Knn_abalone.evaluate(X_train_abalone, y_train_abalone)
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
evaluate_test = clf_Knn_iris.evaluate(X_test_iris, y_test_iris)
for e in evaluate_test:
    print(f'{e}\n {evaluate_test[e]}')

print('---------- WINE TEST ----------')
evaluate_test = clf_Knn_wine.evaluate(X_test_wine, y_test_wine)
for e in evaluate_test:
    print(f'{e}\n {evaluate_test[e]}')

print('---------- ABALONE TEST ----------')
evaluate_test = clf_Knn_abalone.evaluate(X_test_abalone, y_test_abalone)
for e in evaluate_test:
    print(f'{e}\n {evaluate_test[e]}')

# SKLEARN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

neigh_iris = KNeighborsClassifier(n_neighbors=K_opti_iris)
neigh_wine = KNeighborsClassifier(n_neighbors=K_opti_wine)
neigh_abalone = KNeighborsClassifier(n_neighbors=K_opti_abalone)

neigh_iris.fit(X_train_iris, y_train_iris)
neigh_wine.fit(X_train_wine, y_train_wine)
neigh_abalone.fit(X_train_abalone, y_train_abalone)

y_pred_iris = neigh_iris.predict(X_test_iris)
score_iris = neigh_iris.score(X_test_iris, y_test_iris)
print(f'sklearn matrice de confusion iris : \n {confusion_matrix(y_test_iris, y_pred_iris)}')
print(f'sklearn précision iris = {score_iris}')

y_pred_wine = neigh_wine.predict(X_test_wine)
score_wine = neigh_wine.score(X_test_wine, y_test_wine)
print(f'sklearn matrice de confusion iris : \n {confusion_matrix(y_test_wine, y_pred_wine)}')
print(f'sklearn précision iris = {score_wine}')

y_pred_abalone = neigh_abalone.predict(X_test_abalone)
score_abalone = neigh_abalone.score(X_test_abalone, y_test_abalone)
print(f'sklearn matrice de confusion iris : \n {confusion_matrix(y_test_abalone, y_pred_abalone)}')
print(f'sklearn précision iris = {score_abalone}')


###############################################################################################
#                                        Naives Bayes                                         #
###############################################################################################
print("\n\n ------- NAIVES BAYES -------\n\n")

# Charger/lire les datasets, reset à 0 car modifcations lors de KNN
X_train_iris, y_train_iris, X_test_iris, y_test_iris = load_iris_dataset(train_ratio=0.7, seed=69, shuffle=True)
X_train_wine, y_train_wine, X_test_wine, y_test_wine = load_wine_dataset(train_ratio=0.7, seed=42, shuffle=False)
X_train_abalone, y_train_abalone, X_test_abalone, y_test_abalone = load_abalone_dataset(train_ratio=0.7, seed=42, shuffle=False)

# Encode Iris y
le = labelEncoder()
y_train_iris = le.fit_transform(y_train_iris)
y_test_iris = le.transform(y_test_iris)

# Initialiser et entrainer les classifieurs
clf_nb_iris = cv_NaivesBayes_init_train(X_train_iris, y_train_iris)
clf_nb_abalone = cv_NaivesBayes_init_train(X_train_abalone, y_train_abalone)
clf_nb_wine = cv_NaivesBayes_init_train(X_train_wine, y_train_wine)

# Evaluer le classifieur sur les données de train
evaluation_nb_iris_train = cv_NaivesBayes_evaluate(clf_nb_iris, X_train_iris, y_train_iris)
evaluation_nb_abalone_train = cv_NaivesBayes_evaluate(clf_nb_abalone, X_train_abalone, y_train_abalone)
evaluation_nb_wine_train = cv_NaivesBayes_evaluate(clf_nb_wine, X_train_wine, y_train_wine)

# Evaluer le classifieur sur les données de test
evaluation_nb_iris = cv_NaivesBayes_evaluate(clf_nb_iris, X_test_iris, y_test_iris)
evaluation_nb_abalone = cv_NaivesBayes_evaluate(clf_nb_abalone, X_test_abalone, y_test_abalone)
evaluation_nb_wine = cv_NaivesBayes_evaluate(clf_nb_wine, X_test_wine, y_test_wine)

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
# Affiche résultats pour train
print('---------- IRIS TRAIN ----------')
for e in evaluation_nb_iris_train:
    print(f'{e}\n {evaluation_nb_iris_train[e]}')

print('---------- WINE TRAIN ----------')
for e in evaluation_nb_wine_train:
    print(f'{e}\n {evaluation_nb_wine_train[e]}')

print('---------- ABALONE TRAIN ----------')
for e in evaluation_nb_abalone_train:
    print(f'{e}\n {evaluation_nb_abalone_train[e]}')


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
# Affiche résultats pour train
print('---------- IRIS TEST ----------')
for e in evaluation_nb_iris:
    print(f'{e}\n {evaluation_nb_iris[e]}')

print('---------- WINE TEST ----------')
for e in evaluation_nb_wine:
    print(f'{e}\n {evaluation_nb_wine[e]}')

print('---------- ABALONE TEST ----------')
for e in evaluation_nb_abalone:
    print(f'{e}\n {evaluation_nb_abalone[e]}')


print("\n\nSKLEARN GaussianNB\n\n")

# SKLEARN

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def mean_precision(cf):
    precisions = []
    for i, row in enumerate(cf):
        precisions.append(row[i] / np.sum(row))
    return np.mean(precisions)

def mean_recall(cf):
    recalls = []
    for i, row in enumerate(np.transpose(cf)):
        recalls.append(row[i] / np.sum(row))
    return np.mean(recalls)

def mean_F1_score(precision, recall):

    F1_scores = ((2 * precision * recall) / (precision + recall))
    return F1_scores

def printScores(cf, y_pred):
    accuracy = np.trace(cf) / len(y_pred)
    precision = mean_precision(cf)
    recall = mean_recall(cf)
    F1_score = mean_F1_score(precision, recall)
    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("Rappel : ", recall)
    print("Score F1 : ", F1_score)

clf_nb2_iris = GaussianNB()
clf_nb2_wine = GaussianNB()
clf_nb2_abalone = GaussianNB()

clf_nb2_iris.fit(X_train_iris, y_train_iris)
clf_nb2_wine.fit(X_train_wine, y_train_wine)
clf_nb2_abalone.fit(X_train_abalone, y_train_abalone)

y_pred2_iris = clf_nb2_iris.predict(X_test_iris)
y_pred2_wine = clf_nb2_wine.predict(X_test_wine)
y_pred2_abalone = clf_nb2_abalone.predict(X_test_abalone)

score2_iris = clf_nb2_iris.score(X_test_iris, y_test_iris)
cf = confusion_matrix(y_test_iris, y_pred2_iris)
print("\n IRIS \n")
print(f'sklearn matrice de confusion Iris : \n {cf}')
printScores(cf, y_pred2_iris)

score2_wine = clf_nb2_wine.score(X_test_wine, y_test_wine)
cf = confusion_matrix(y_test_wine, y_pred2_wine)
print("\n\n WINE \n")
print(f'sklearn matrice de confusion Wine : \n {cf}')
printScores(cf, y_pred2_wine)

score2_abalone = clf_nb2_abalone.score(X_test_abalone, y_test_abalone)
cf = confusion_matrix(y_test_abalone, y_pred2_abalone)
print("\n\n ABALONE \n")
print(f'sklearn matrice de confusion Abalone : \n {cf}')
printScores(cf, y_pred2_abalone)

