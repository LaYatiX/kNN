from unittest import TestCase
from KNN import KNN
import numpy as np
import pandas





class KNNTest(TestCase):
 
    def setUp(self):
        data = pandas.read_csv("iris.data.learning", header=None)
        test = pandas.read_csv("iris.data.test", header=None)
        trainingData = np.array(data)
        testData = np.array(test)
        testValues = []
        testLabels = []
        for record in testData:
            testValues.append(record[0:4])
            testLabels.append(record[4])
        self.kNN =  KNN(5, trainingData)

    def testDistance(self):
        self.assertEqual(self.kNN.distance([0],[3]), 3)
        self.assertEqual(self.kNN.distance([0, 5],[0,3]), 2)


    
    def testScore(self):
        exampleData = [
            [4.9,3.1,1.5,0.1],
            [6.6,2.9,4.6,1.3],
            [4.4,3.2,1.3,0.2],
            [5.5,2.3,4.0,1.3],
            [5.1,3.4,1.5,0.2],
            [6.8,2.8,4.8,1.4]
        ]
        exampleResult=['Iris-setosa', 'Iris-versicolor','Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-versicolor']
        self.assertEqual(self.kNN.score(exampleData, exampleResult), 1)
    
    def testPredict(self):
        exampleData = [
            [4.9,3.1,1.5,0.1],
            [6.6,2.9,4.6,1.3],
            [4.4,3.2,1.3,0.2],
            [5.5,2.3,4.0,1.3],
            [5.1,3.4,1.5,0.2],
            [6.8,2.8,4.8,1.4]
        ]
        exampleResult=['Iris-setosa', 'Iris-versicolor','Iris-setosa', 'Iris-versicolor', 'Iris-setosa', 'Iris-versicolor']
        self.assertEqual(self.kNN.predict(exampleData),exampleResult)

