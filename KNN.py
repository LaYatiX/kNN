import operator
from scipy.spatial import distance
from data import Data

class KNN:
    k = 0
    model = []
    size = 0
    def __init__(self, k, trainingData):
        self.k = k
        for x in trainingData:
            self.model.append(Data(x))
        self.size = len(self.model)

    # przyjmuje listę danych bez etykiet
    # zwraca listę etykiet dla zestawu danych bez etykiet
    def predict(self, unlabeledData):  
        result = []
        for record in unlabeledData:
            distancesAndLabels = []
            nearestNeighboursLabels = []
            for v in self.model:
                distancesAndLabels.append([self.distance(record, v.vector), v.label])
            distancesAndLabels.sort(key=lambda x: x[0])  # sortuje po dystansie między wektorami

            # dodaj etykiety k najblizszych sąsiadów do listy
            for n in range(self.k):
                nearestNeighboursLabels.append(distancesAndLabels[n][1])  
            #dodaj najczęściej wystepującą etykietę do listy wyników
            result.append(max(nearestNeighboursLabels, key=nearestNeighboursLabels.count))
        return result

    # przyjmuje listę danych bez etykiet oraz liste poprawnych etykiet dla tych danych
    # zwraca stopień poprawności predykcji w skali od 0 - 1
    def score(self, unlabelledData, labels):  # zwraca współczynnik poprawnie rozpoznanych obiektów.
        predictedLabels = self.predict(unlabelledData)
        i = 0
        same = 0
        while i < len(labels):
            if predictedLabels[i] == labels[i]:
                same = same + 1
            i = i + 1
            
        return same/len(labels)

    def distance(self, v1, v2):  # zwraca długość pomiedzy wektorami
        return distance.euclidean(v1, v2)
