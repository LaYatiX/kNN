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
        self.assertEqual(self.kNN.distance([0, 5],[3]), 3)
        self.assertEqual(self.kNN.distance([0, 5, 4],[3, 3, 5]), 3)
        self.assertEqual(self.kNN.distance([0],[3]), 3)
        self.assertEqual(self.kNN.distance([0],[3]), 3)


    
    def testScore(self):
        pass
    
    def testPredict(self):
        pass

