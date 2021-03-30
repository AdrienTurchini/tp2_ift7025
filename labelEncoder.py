import numpy as np

class labelEncoder:
    
    def fit(self, y):
        self.labels = list(np.unique(y))

    def transform(self, y):
        y_encoded = []
        for yi in y:
            y_encoded.append(self.labels.index(yi))
        return np.array(y_encoded)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_encoded):
        y = []
        for yi in y_encoded:
            y.append(self.labels[yi])
        return np.array(y)