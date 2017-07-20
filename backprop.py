

import numpy as np
import numpy.matlib


# layer sigmoid
def sigm(x):
    '''Sigmoide function '''
    return 1/(1 + np.exp(-x));

def dsigmdx(x,dzdx):
    '''dSigm/dx derivative of the sigmoide function'''
    return sigm(x)*(1-sigm(x))*dzdx;
#------

# layer softmat
def softmax(x):
    """Soft function"""
    return np.exp(x) / np.sum(np.exp(x), axis = 0) 

def dsoftmatdx(x,dzdx):
    '''dSigm/dx derivative of the sigmoide function'''
    return softmax(x)*(1-softmax(x))*dzdx;
#------


# layer inner
def inner(x,w):
    ''' 
    z=W*x+b 
    note: b = 0
    '''
    return  np.dot(w,x);

def dinnerdx(x,w, dzdx):
    '''
    dinnerdx derivate with respect to x
    '''
    return np.dot(w.T, dzdx);

def dinnerdw(x,w, dzdx):
    '''
    dinnerdw derivate with respect to w
    '''
    return np.dot(dzdx,x.T);
# ---

# layer loss
def loss(x,y):
    ''' Loss function 
        loss = (1/2)(y-x)^2
    '''
    return (1/(2*len(y))*np.sum((y-x)**2));


def dlossdx(x,y, dzdx):
    '''
    dlossdx derivate with respect to x
    Ecuation: d/dx (1/n)\sum_x (y-x)^2
    '''
    return (x-y)*dzdx
# ---


#       inner           inner            sigm                   loss
# z1-->|w1x+b|--(z2)-->|w2x+b|--(z3)-->|sigm(x)|--(z4)-->|E=1/n\sum((y-x)^2)|

def forward( x, w1, w2):
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
    z2 = inner(z1,w1); 
    z3 = inner(z2,w2);      
    z4 = sigm(z3);  
    return z4;


def backward(x, y, w1, w2):
    '''
    backward function
    Compute dervative with respect to w1 and w2
    '''

    # forward --->
    z1 = x;
    z2 = inner(z1,w1);
    z3 = inner(z2,w2);
    z4 = sigm(z3);

    E = loss(z4,y);

    # <--- backward
    l5 = 1;
    l4 = dlossdx(z4, y, l5)    
    l3 = dsigmdx(z3,l4);
    l2 = dinnerdx(z2,w2,l3); dEdW2 = dinnerdw(z2,w2,l3);  
    l1 = dinnerdx(z1,w1,l2); dEdW1 = dinnerdw(z1,w1,l2);
   
    # note: l1 not have need to calculated

    return E, dEdW1, dEdW2



# data
x = np.matrix([[1,2,3]]).T;
y = np.matrix([1.0]);

# init weigth
w1 = np.matrix([[0.1, 0.2, 0.3],[0.1, 0.2, 0.3]])
w2 = np.matrix([0.8, 0.2])

# Function to minimize
# J(x)_{w1,w2}
y_ = forward(x,w1,w2);
e = loss(y_,y);

# derivate
# grad J(x)
E, dEdW1, dEdW2 = backward(x, y, w1, w2);


# minimization with gradien decent
# w^t = w^(t-1) + lr*gardJ



print(E)
print(dEdW1)
print(dEdW2)



print('ok!!!!')