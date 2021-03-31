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
        return (((x1 - x2)**2).sum())**0.5

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
        self.n_class = len(np.unique(train_labels))

    def getNeighbors(self, X_test_row):
        '''
        Description:
            Calcul les distances avec X_test_row
        Output:
            k-nearest neighbors to the test data
        '''
        distances = []
        for X_train_row, y_train_row in zip(self.X_train, self.y_train):
            distances.append([X_train_row, self.euclidean_distance(X_test_row, X_train_row), y_train_row])
        distances.sort(key=lambda distances: distances[1])

        K_neighbors = []
        for i in range(self.K):
            K_neighbors.append(distances[i])
        
        return K_neighbors


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

    def confusion_matrix(self, y_pred, y):
        """
        retourne la matrice de confusion
        """
        cf = np.zeros((3, 3), dtype='int64')
        for yi, yhat in zip(y, y_pred):
            for i in range(self.n_class):
                for j in range(self.n_class):
                    if yi == i and yhat == j:
                        cf[i, j] += 1
        return cf

    def mean_precision(self, cf):
        self.precisions = []
        for i, row in enumerate(cf):
            self.precisions.append(row[i] / np.sum(row))
        return np.mean(self.precisions)

    def mean_recall(self, cf):
        self.recalls = []
        for i, row in enumerate(np.transpose(cf)):
            self.recalls.append(row[i] / np.sum(row))
        return np.mean(self.recalls)

    def mean_F1_score(self):
        assert len(self.precisions) and len(self.recalls)
        
        self.F1_scores = []
        for precision, recall in zip(self.precisions, self.recalls):
            self.F1_scores.append((2 * precision * recall) / (precision + recall))
        return np.mean(self.F1_scores)

    def evaluate(self, X, y):
        """
        c'est la méthode qui va evaluer votre modèle sur les données X
        l'argument X est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        y : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        F1 = (2 * self.precision() * self.recall()) / (self.precision() + self.recall())

        """
        y_pred = self.predict(X)
        cf = self.confusion_matrix(y_pred, y)
        
        # Metrics
        accuracy = np.trace(cf) / len(y)
        precision = self.mean_precision(cf)
        recall = self.mean_recall(cf)
        F1_score = self.mean_F1_score()
            
        return {'mean_accuracy': accuracy, 
                'mean_precision': precision, 
                'mean_recall': recall, 
                'mean_F1-score': F1_score, 
                'Confusion_matrix': cf
                }