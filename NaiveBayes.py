"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
    * train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
    * predict 	: pour prédire la classe d'un exemple donné.
    * evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class NaiveBayes: #nom de la class à changer

    def __init__(self, **kwargs):
        """
        C'est un Initializer. 
        Vous pouvez passer d'autre paramètres au besoin,
        c'est à vous d'utiliser vos propres notations
        """
        
        
    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        """
        C'est la méthode qui va entrainer votre modèle,
        train est une matrice de type Numpy et de taille nxm, avec 
        n : le nombre d'exemple d'entrainement dans le dataset
        m : le mobre d'attribus (le nombre de caractéristiques)

        train_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        """
        self.classes, count = np.unique(train_labels, return_counts=True) # label des classes, nombres de données par classe
        self.n = train.shape[0] # le nombre d'exemple d'entrainement dans le dataset
        self.m = train.shape[1] # le mobre d'attribus

        freq_byClass = [] # tableau des fréquences de chaque classe parmis la totalité des données
        for nb in count:
            freq_byClass.append(nb/self.n)
        
        data_byClass = [] # tableau des sous-datasets selon chaque classe 
        for class_i in self.classes:
            actualClass = []
            for X_i, y_i in zip(train, train_labels):
                if y_i == class_i:
                    actualClass.append(X_i)
            data_byClass.append(actualClass)

        self.mean_byClass = [] # tableau de tableau des moyennes de chaque attribut pour chaque classe
        self.var_byClass = [] # tableau de tableau variances de chaque attribut pour chaque classe
        for data_i in data_byClass:
            mean = np.mean(data_i, axis = 0)
            self.mean_byClass.append(mean)
            var = np.var(data_i, axis = 0)
            self.var_byClass.append(var)

    # Calcul proba loi normale
    def calculProba(self, X, mean, var):
        if var == 0: # si variance nulle, var = 1 pour éviter division par 0
            var = 1 
        gaussienne = (1/(np.sqrt(2 * np.pi * var))) * np.exp((-1/(2*var)) * np.square(X - mean))
        return gaussienne
            
    def predict(self, x):
        """
        Prédire la classe d'un exemple x donné en entrée
        exemple est de taille 1xm
        """
        probaPosteriori = [] # proba a posteriori de chaque classe
        for i in range(len(self.classes)):
            mean = self.mean_byClass[i][0]
            var = self.var_byClass[i][0]
            probaClass_i = self.freq_byClass[i] # probabilité de la classe i
            for i, (meanAttribute, varAttribute) in enumerate(zip(mean, var)):
                probaConditionelle = self.calculProba(x[i], meanAttribute, varAttribute)
                probaClass_i *= probaConditionelle
            probaPosteriori.append(probaClass_i)

        probaPosterioriMax = np.max(probaPosteriori) # on cherche la probabilité à poseteriori maximale
        classToReturn = self.classes[probaPosteriori.index(probaPosterioriMax)] # on cherche la classe associée à l'index de la proba a posteriori maximale

        return classToReturn
        
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
        
    
    # Vous pouvez rajouter d'autres méthodes et fonctions,
    # il suffit juste de les commenter.