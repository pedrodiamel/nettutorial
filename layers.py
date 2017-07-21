

import numpy as np
import numpy.matlib

# Layer class 
class layer(object):
    
    def forward(self, param):
        raise NotImplementedError

    def backward(self, param):
        raise NotImplementedError

# Inner layer
class inner(layer):

    def forward(self, param): 
        return np.dot(param['w'],param['x']);

    def backward(self, param):
        return np.dot(param['w'].T, param['dzdx']), np.dot(param['dzdx'],param['x'].T); 

# Sigmoid layer
class sigm(layer):
    
    def forward(self, param): 
        return self._sigm(param['x']);

    def backward(self, param):
        return self._sigm(param['x'])*(1-self._sigm(param['x']))*param['dzdx']; 

    def _sigm(self, x): return 1/(1 + np.exp(-x));


# Loss layer
class loss(layer):
    
    def forward(self, param): 
        return  (1/(2*len(param['y']))*np.sum((param['y']-param['x'])**2));

    def backward(self, param):
        return (param['x']-param['y'])*param['dzdx']; 




# Const function
def costFunc( x, w1, w2):
    '''
    forward function
    Entrada:
        * x vector nxm. m featurs
        * w1, w2 weigths
    Return:
        * z4
    
    Note: b=0
    '''

    z1 = x;
    z2 = inner().forward({'x':z1, 'w':w1}) 
    z3 = inner().forward({'x':z2, 'w':w2}) 
    z4 =  sigm().forward({'x':z3});
    return z4;


# Gradind function
def gradCostFunc(x, y, w1, w2):
    '''
    Apply backward function
    Compute dervative with respect to w1 and w2
    '''
    
    # forward --->
    z1 = x;
    z2 = inner().forward({'x':z1, 'w':w1}) 
    z3 = inner().forward({'x':z2, 'w':w2}) 
    z4 =  sigm().forward({'x':z3});
    z5 = loss().forward({'x':z4, 'y':y});

    E = z5;

    # <--- backward
    l5 = 1;
    l4 = loss().backward({'x':z4, 'y':y, 'dzdx':l5} );    
    l3 = sigm().backward({'x':z3, 'dzdx':l4});
    l2, dEdW2 = inner().backward({'x':z2, 'w':w2, 'dzdx':l3});
    _ , dEdW1 = inner().backward({'x':z1, 'w':w1, 'dzdx':l2});


    return E, dEdW1, dEdW2


