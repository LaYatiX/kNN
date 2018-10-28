import operator
from scipy.spatial import distance
from data import Data

class KNN:
    k = 0
    model = []
    size = 0
    def __init__(self, k, inputData):
        self.k = k
        for x in inputData:
            self.model.append(Data(x))
        self.size = len(self.model)

    def predict(self, unlabeledData):  # zwraca listę rozpoznanych etykiet
        result = []
        for record in unlabeledData:
            distances = []
            labels = []
            for v in self.model:
                distances.append([self.distance(record, v.vector), v.label])
            distances.sort(key=lambda x: x[0])  # sortuje po dystansie między wektorami
            for n in range(self.k):
                labels.append(distances[n][1])  # dodaje etykiety do końcowego wyniku
            
        return result

    def score(self, data, labels):  # zwraca współczynnik poprawnie rozpoznanych obiektów.
        return 0

    def distance(self, v1, v2):  # zwraca długość pomiedzy wektorami
        return distance.euclidean(v1, v2)
