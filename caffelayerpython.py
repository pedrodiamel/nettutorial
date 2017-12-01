
import caffe
import numpy as np
from skimage import feature
from skimage import data, color, exposure
import scipy.misc

class pydatagenerate(caffe.Layer):
    
    def setup(self, bottom, top):
        
        # Check top shape
        if len(top) != 2: raise Exception("Need to define tops (data, label)")
        
        #Check bottom shape
        if len(bottom) != 0: raise Exception("Do not define a bottom.")
        
        #Read parameters
        params = eval(self.param_str)
        self.batch_size = params["batch_size"]
        self.im_shape = params["im_shape"]
        self.mu = params["mu"]


        #Reshape top        
        top[0].reshape(self.batch_size, 1, self.im_shape, self.im_shape)
        top[1].reshape(self.batch_size, 1, self.im_shape, self.im_shape)
               
        
    def forward(self, bottom, top):
        
        for itt in range(self.batch_size):
           
            # Use the batch loader to load the next image.
            im, label = self.load_next_img()
            
            #Here we could preprocess the image
            # ...
            interp = 'bilinear';
            im = scipy.misc.imresize(im, (self.im_shape, self.im_shape), interp=interp);
            label = scipy.misc.imresize(label, (self.im_shape, self.im_shape), interp=interp);

            # Add directly to the top blob
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
    
    
    def load_next_img(self):
        
        # get image dummy
        image = color.rgb2gray(data.astronaut())

        # get label        
        label = (image < self.mu);        
        return image, label
    
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (img shape and batch size)
        """
        pass

    def backward(self, bottom, top):
        """
        This layer does not back propagate
        """
        pass




class pyimgloss(caffe.Layer):
    

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2: raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        
        # check input dimensions match
        # if bottom[0].count != bottom[1].count: raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        
        # loss output is scalar
        top[0].reshape(1)
    

    def forward(self, bottom, top):
        '''
             1                    1  
        E = ---\sum max{0,1-x} + --- \sum max{0,x}
            |P| xEP              |N|   xEN
        '''
         
        x = bottom[0].data;
        y = bottom[1].data;

        pos = (np.array(y)==1);
        neg = (np.array(y)!=1);

        #   E(1,t) = ...
        # mean(max(0, 1 - res.x3(pos))) + ...
        # mean(max(0, res.x3(neg))) ;

        E = np.mean( self._amax(1-x[pos]) ) + np.mean( self._amax(x[neg]) );
        top[0].data[...] = E;

    def backward(self, top, propagate_down, bottom):
        
        # dzdx3 = ...
        #     - single(res.x3 < 1 & pos) / sum(pos(:)) + ...
        #     + single(res.x3 > 0 & neg) / sum(neg(:)) ;

        x = bottom[0].data;
        y = bottom[1].data;
        
        pos = (np.array(y)==1);
        neg = (np.array(y)!=1);

        dzdx = -( x<1 and pos) / np.sum(pos) + ( x>1 and nos) / np.sum(nos);  

        bottom[0].diff[...] =  dzdx;
        bottom[1].diff[...] = -dzdx;


    def _amax(self, x): return x*(x>0) if x.tolist() else 0.0;

