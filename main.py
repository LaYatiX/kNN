import numpy as np
from scipy.spatial import distance
import pandas
import operator

class Data:  # Klasa do przechowywania pojedynczej danej z pliku
    vector1 = [0, 0, 0, 0]
    label = ""

    def __init__(self, data):
        self.vector = [data[0], data[1], data[2], data[3]]
        self.label = data[4]


class EuklidesToSpokoGosc:
    k = 0
    data = []
    size = 0
    def __init__(self, k, data):
        self.k = k
        for x in data:
            self.data.append(Data(x))
        self.size = len(self.data)

    def predict(self, data):  # zwraca listę rozpoznanych etykiet
        result = []
        for record in data:
            distances = []
            labels = []
            for v in self.data:
                distances.append([self.distance(record, v.vector), v.label])
            # distances = sorted(distances, key=lambda x: x[0])
            distances.sort(key=lambda x: x[0])  # sortuje po dystansie między wektorami
            for n in range(self.k):
                labels.append(distances[n][1])  # dodaje etykiety do końcowego wyniku
            result.append(labels)
        return result

    def score(self, data, labels):  # zwraca współczynnik poprawnie rozpoznanych obiektów.
        return 0

    def distance(self, v1, v2):  # zwraca długość pomiedzy wektorami
        return distance.euclidean(v1, v2)


data = pandas.read_csv("iris.data.learning")
test = pandas.read_csv("iris.data.test")
a = np.array(data)
print(a)  # wypisuje dane z pliku
kNN = EuklidesToSpokoGosc(5, a)

print(kNN.data[0].vector)

distance2 = distance.euclidean(kNN.data[0].vector, kNN.data[1].vector)
print(kNN.distance(0, 1))
print(kNN.size)

# print()

print(np.matrix(kNN.predict([[5.9, 3.1, 5.0, 1.8], [4.4, 3.2, 1.3, 0.2], [6.0,2.7,5.1,1.6], [6.3,2.8,5.1,1.5], [6.3,2.5,5.0,1.9], [4.9,3.1,1.5,0.1]])))
# 4.4,3.2,1.3,0.2


