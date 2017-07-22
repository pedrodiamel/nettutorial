

import numpy as np
import matplotlib.pyplot as plt


def amax(x): return x*(x>0);

#i = np.array([[1,0,1],[1,0,1]])==1;
# p = np.array([[1,-2,3],[4,5,-6]]);

x = np.random.randn(100,100);
neg = x <  0.0;
pos = x >= 0.0;

#   E(1,t) = ...
# mean(max(0, 1 - res.x3(pos))) + ...
# mean(max(0, res.x3(neg))) ;

# plt.imshow(x)
# plt.show()

E = np.mean( amax(1-x[pos]) ) + np.mean( amax(x[neg]) );

print( E )
 
