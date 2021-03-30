import numpy as np

class Knn:

    def __init__(self, K=5):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.K = K

    def euclidean_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1) - 1):
            distance += (x1[i] - x2[i])**2
        return distance**0.5

    def train(self, train, train_labels):  # vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        train_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        """
        assert len(train) == len(train_labels)
        self.X_train = train
        self.y_train = train_labels

    def getNeighbors(self, X_test_row):
        '''
        Description:
            Calcul les distances avec X_test_row
        Output:
            k-nearest neighbors to the test data
        '''
        distances = []
        for i, X_train_row in enumerate(self.X_train):
            distances.append([X_train_row, self.euclidean_distance(X_test_row, X_train_row), self.y_train[i]])
        distances.sort(key=lambda distances: distances[1])

        neighbors = []
        for i in range(self.K):
            neighbors.append(distances[i])
        return neighbors


    def predict(self, X):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        y_pred = []
        
        for X_test_row in X:
            K_neighbors = self.getNeighbors(X_test_row)
            classes = [row[-1] for row in K_neighbors]
            prediction = max(set(classes), key=classes.count)
            y_pred.append(prediction)
        
        return np.array(y_pred)

    def accuracy(self):
        """
        Calcul l'accuracy avec :
        ACC = (TP + TN) / (TP + TN + FP + FN)
        """
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def precision(self):
        """
        Calcul la precision avec :
        PPV = (TP) / (TP + FP)
        """
        return (self.TP) / (self.TP + self.FP)

    def recall(self):
        """
        Calcul le recall avec :
        TPR = (TP) / (TP + FN)
        """
        return (self.TP) / (self.TP + self.FN)

    def f1_score(self):
        """
        Calcul le f1_score avec :
        ACC = 2 * (PPV * TPR) / (PPV + TPR)
        """
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())

    def confusion_matrix(self):
        """
        retourne la matrice de confusion
        """
        return np.array([[self.TP, self.FP], [self.FN, self.TN]])

    def evaluate(self, X, y):
        """
        c'est la méthode qui va evaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        y : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire
        """
        y_pred = self.predict(X)

        self.TP = np.sum(y[y==1] == y_pred[y==1])
        self.TN = np.sum(y[y==0] == y_pred[y==0])
        self.FP = np.sum(y[y==0] != y_pred[y==0])
        self.FN = np.sum(y[y==1] != y_pred[y==1])

        return {'Accuracy': self.accuracy(), 
                'Precision': self.precision(), 
                'Recall': self.recall(), 
                'F1-score': self.f1_score(), 
                'Confusion_matrix': self.confusion_matrix()
                }