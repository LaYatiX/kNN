import numpy as np
import pandas
from KNN import KNN




data = pandas.read_csv("iris.data.learning")
test = pandas.read_csv("iris.data.test")
a = np.array(data)
print('dane')
print(a)  # wypisuje dane z pliku
kNN = KNN(5, a)


print('wektor')

print(kNN.model[0].vector)

print('size')
print(kNN.size)

# print()

print(np.matrix(kNN.predict([[5.9, 3.1, 5.0, 1.8], [4.4, 3.2, 1.3, 0.2], [6.0,2.7,5.1,1.6], [6.3,2.8,5.1,1.5], [6.3,2.5,5.0,1.9], [4.9,3.1,1.5,0.1]])))
# 4.4,3.2,1.3,0.2


