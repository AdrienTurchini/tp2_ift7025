import numpy as np

X = ([2, 4, 2], [3, 2, 1], [5, 3, 4], [43, 4, 2], [2, 4, 43], [25, 43, 2], [2, 42, 24], [2, 4, 2])
X = np.matrix(X)
#X = np.array(X)
y = ["iris", "setosa", "iris", "iris", "iris", "setosa", "setosa", "aray"]

data = [100, 0, 3]


class  NaiveBayes:

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
        # classe = label des classes
        # count = nombre de données par classes
        self.classes, count = np.unique(train_labels, return_counts=True)
        self.n = train.shape[0] # a finir
        self.m = train.shape[1] # a finir
        self.freq_byClass = []
        for nb in count:
            self.freq_byClass.append(nb/self.n)
        
        # pour chaque classe on créer un sous dataset de la classe contenu dans le tab dataSplit
        data_byClass = []
        for class_i in self.classes:
            actualClass = []
            for X_i, y_i in zip(train, train_labels):
                if y_i == class_i:
                    actualClass.append(X_i)
            data_byClass.append(actualClass)

        self.mean_byClass = []
        self.var_byClass = []

        for data_i in data_byClass:
            mean = np.mean(data_i, axis = 0)
            self.mean_byClass.append(mean)
            var = np.var(data_i, axis = 0)
            self.var_byClass.append(var)


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

        print(probaPosteriori)
        probaPosterioriMax = np.max(probaPosteriori) # on cherche la probabilité à poseteriori maximale
        classToReturn = self.classes[probaPosteriori.index(probaPosterioriMax)] 

        print(classToReturn)
            


        


clf = NaiveBayes()
clf.train(train = X, train_labels= y)
clf.predict(data)