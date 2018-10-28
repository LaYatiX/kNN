import numpy as np
import pandas
from KNN import KNN




data = pandas.read_csv("iris.data.learning", header=None)
test = pandas.read_csv("iris.data.test", header=None)
trainingData = np.array(data)
testData = np.array(test)
testValues = []
testLabels = []
for record in testData:
    testValues.append(record[0:4])
    testLabels.append(record[4])

    
#training
kNN = KNN(5, trainingData)

#print  test values
print("Wartosci do testu")
for value in testValues:
    print(value)
#print labels for test data
print("Uzyskane etykiety")
print(kNN.predict(testValues))
#print score for test data
print("Stopien poprawnosci")
print(kNN.score(testValues, testLabels))






