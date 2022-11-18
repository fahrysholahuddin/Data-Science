from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

class class_NB:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)
        print("Jumlah data :", len(self.y))

    def model(self):
        print("X_train in model NB :\n", self.X)
        print("Panjang data X_train : ",len(self.X))
        self.model = GaussianNB()
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        print("prediksi model NB: ", self.model.predict(data_testing))
        print("=======================================")
        return self.model.predict(data_testing)
class class_KNN:
    X = []
    y = []
    model = None

    def __init__(self, data, label):
        self.X = data
        self.y = label

    def viewDataset(self):
        print("data :", self.X)
        print("label :", self.y)
        print("Jumlah data :", len(self.y))

    def model(self):
        print("X_train in model KNN :\n", self.X)
        print("Panjang data X_train : ",len(self.X))
        self.model = KNeighborsClassifier(n_neighbors=10)
        self.model.fit(np.array(self.X), self.y)

    def predict(self, data_testing):
        print("prediksi model KNN : ", self.model.predict(data_testing))
        print("=======================================")
        return self.model.predict(data_testing)