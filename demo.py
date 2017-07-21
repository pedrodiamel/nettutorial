



import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import layers as net



# def plotData(x,y):
#     plt.plot(x,y,'or');
    # plt.ylabel('Profit in $10,000s');
    # plt.xlabel('Population of City in 10,000s');
    # plt.grid(1)



# Load data
data = np.loadtxt('data.txt', delimiter=',');
X = data[:,0]; y = data[:,1];
m = len(y);

# plot data
# plotData(X,y);



# print(data)









# # data
# x = np.matrix([[1,2,3]]).T;
# y = np.matrix([5.0]);

# # init weigth
# w1 = np.matrix([[0.1, 0.2, 0.3],[0.1, 0.2, 0.3]])
# w2 = np.matrix([0.8, 0.2])


# y_ = costFunc( x, w1, w2);
# e = loss().forward({'x':y_,'y':y});
# print(e)

# # derivate
# # grad J(x)
# E, dEdW1, dEdW2 = gradCostFunc(x, y, w1, w2);


# # minimization with gradien decent
# # w^t = w^(t-1) + lr*gardJ

# print(E)
# print(dEdW1)
# print(dEdW2)


print('ok!!!!')