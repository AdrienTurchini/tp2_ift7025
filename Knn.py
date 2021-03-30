import numpy as np

class Knn:

    def __init__(self, K=5):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        self.K = K

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

    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        return np.array([])

    def accuracy(self):
        """
        ACC = (TP + TN) / (TP + TN + FP + FN)
        """
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def precision(self):
        """
        PPV = (TP) / (TP + FP)
        """
        return (self.TP) / (self.TP + self.FP)

    def recall(self):
        """
        TPR = (TP) / (TP + FN)
        """
        return (self.TP) / (self.TP + self.FN)

    def f1_score(self):
        """
        ACC = 2 * (PPV * TPR) / (PPV + TPR)
        """
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())

    def confusion_matrix(self):
        """
        Compute the confusion matrix
        """
        return np.array([self.TP, self.FP], [self.FN, self.TN])

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
        self.FN = np.sum(y[y==1] != y_pred[y==1])
        self.FP = np.sum(y[y==0] != y_pred[y==0])

        return {'Accuracy': self.accuracy(), 
                  'Precision': self.precision(), 
                  'Recall': self.recall(), 
                  'F1-score': self.f1_score(), 
                  'Confusion_matrix': self.confusion_matrix()
                  }


    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter.
