import operator
from scipy.spatial import distance
from data import Data

class KNN:
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
            distances.sort(key=lambda x: x[0])  # sortuje po dystansie między wektorami
            for n in range(self.k):
                labels.append(distances[n][1])  # dodaje etykiety do końcowego wyniku
            result.append(labels)
        return result

    def score(self, data, labels):  # zwraca współczynnik poprawnie rozpoznanych obiektów.
        return 0

    def distance(self, v1, v2):  # zwraca długość pomiedzy wektorami
        return distance.euclidean(v1, v2)
