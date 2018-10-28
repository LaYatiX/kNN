import numpy as np
import pandas
from KNN import KNN




data = pandas.read_csv("iris.data.learning")
test = pandas.read_csv("iris.data.test")
trainingData = np.array(data)
testData = np.array(test)
testValues = []
testLabels = []
for record in testData:
    testValues.append(record[0:4])
    testLabels.append(record[4])

kNN = KNN(5, trainingData)
print(kNN.score(testValues, testLabels))






