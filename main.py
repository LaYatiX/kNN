import numpy as np
from scipy.spatial import distance
import pandas


class Data:  # Klasa do przechowywania pojedynczej danej z pliku
    vector1 = [0, 0]
    vector2 = [0, 0]
    label = ""

    def __init__(self, data):
        self.vector1 = [data[0], data[1]]
        self.vector2 = [data[2], data[3]]
        self.label = data[4]


class EuklidesToSpokoGosc:
    k = 0
    data = []

    def __init__(self, k, data):
        self.k = k
        for x in data:
            self.data.append(Data(x))

    def predict(self, data):  # zwraca listę rozpoznanych etykiet
        return []

    def score(self, data, labels):  # zwraca współczynnik poprawnie rozpoznanych obiektów.
        return 0

    def distance(self, record):  # zwraca długość pomiedzy wektorami w jednym rekordzie
        return distance.euclidean(data[record].vector1, data[record].vector2)


data = pandas.read_csv("iris.data.learning")
a = np.array(data)
print(a)  # wypisuje dane z pliku
kNN = EuklidesToSpokoGosc(2, a)

print(kNN.data[0].vector1)

distance2 = distance.euclidean(kNN.data[0].vector1, kNN.data[0].vector2)
print(distance2)
# kNN.pre
