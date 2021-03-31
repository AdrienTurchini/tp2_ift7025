import numpy as np
import random


def train_test_split(X, y, train_size, seed, shuffle):
    assert len(X) == len(y)

    X_shuf, y_shuf = X, y

    if shuffle:
        X_shuf, y_shuf = [], []
        random.seed(seed)
        index = list(range(len(y)))
        random.shuffle(index)
        for i in index:
            X_shuf.append(X[i])
            y_shuf.append(y[i])
        
        X_shuf, y_shuf = np.array(X_shuf), np.array(y_shuf)

    X_train = X_shuf[:int((len(X_shuf)+1)*(train_size))]
    X_test = X_shuf[int((len(X_shuf)+1)*(train_size)):]

    y_train = y_shuf[:int((len(y_shuf)+1)*(train_size))]
    y_test = y_shuf[int((len(y_shuf)+1)*(train_size)):]

    return X_train, y_train, X_test, y_test


def load_iris_dataset(train_ratio=0.7, seed=69, shuffle=True):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    X, y = [], []
    with open('datasets/bezdekIris.data', 'r') as f:
        for line in f:
            X.append(line.rstrip().split(',')[:4])
            y.append(line.rstrip().split(',')[4])
    f.close()

    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='str')

    train, train_labels, test, test_labels = train_test_split(
        X, y, train_ratio, seed, shuffle)

    return train, train_labels, test, test_labels


def load_wine_dataset(train_ratio=0.7, seed=69, shuffle=True):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    X, y = [], []
    with open('datasets/binary-winequality-white.csv', 'r') as f:
        for line in f:
            X.append(line.rstrip().split(',')[:11])
            y.append(line.rstrip().split(',')[11])
    f.close()

    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='int64')

    train, train_labels, test, test_labels = train_test_split(
        X, y, train_ratio, seed, shuffle)

    return train, train_labels, test, test_labels


def load_abalone_dataset(train_ratio=0.7, seed=69, shuffle=True):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    X, y = [], []
    sexes = {'M': 0, 'F': 1, 'I': 2}
    with open('datasets/abalone-intervalles.csv', 'r') as f:
        for line in f:
            line = line.rstrip().split(',')
            line[0] = sexes[line[0]]
            line[8] = float(line[8])
            X.append(line[:8])
            y.append(line[8])
    f.close()

    X = np.array(X, dtype='float64')
    y = np.array(y, dtype='int64')

    train, train_labels, test, test_labels = train_test_split(
        X, y, train_ratio, seed, shuffle)

    return train, train_labels, test, test_labels
