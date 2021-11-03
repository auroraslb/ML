import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns

#Extract data
#X = np.load('Xtrain_Regression_Part1.npy')
#y = np.load('Ytrain_Regression_Part1.npy')

#print(X.head())
# første [] er nedover, andre [] er bortover

#plt.plot(X[:][1] ,y, 'bo')


#for i in range(len(y)):
#    plt.plot(X[:][0],y[i][0])
#    plt.show()



arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('2nd element on 1st dim: ', arr[0, 1])
print('5th element on 2nd dim: ', arr[1, 4])
