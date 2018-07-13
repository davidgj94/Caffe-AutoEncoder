import caffe
import numpy as np
from PIL import Image
import random
from random import shuffle
import skimage.io
import matplotlib.pyplot as plt
from itertools import islice
from pathlib import Path
import random
import pdb

class OneHotLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        labels_dir = params['labels_dir']
        self.seed = int(params['seed'])
        random.seed(self.seed)
        globs = Path(labels_dir).glob('*.png')
        self.indices = ['/' + '/'.join(glob.parts[1:]) for glob in globs]
        shuffle(self.indices)
        self.idx = 0
        if 'binary_mask' in params:
            self.binary_mask = bool(params['binary_mask'])
        else:
            self.binary_mask = False
        
        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define one top")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        #X = Image.open('{}/{}.png'.format(xy_dir,'X'))
        #X = np.array(X, dtype=np.float32)
        #X = (X / 255) * alfa
        
        #Y = Image.open('{}/{}.png'.format(xy_dir,'Y'))
        #Y = np.array(Y, dtype=np.float32)
        #Y = (Y / 192) * alfa
        
        #XY = np.dstack((X,Y))
        #XY = XY.transpose((2,0,1))
        
        #self.batch = np.zeros((batchsize, 2, 192, 256))
        #for idx in range(batchsize):
            #self.batch[idx,:,:,:] = XY


    def reshape(self, bottom, top):
        # reshape tops to fit (leading 1 is for batch dimension)
        self.label = self.load_label(self.indices[self.idx])
        top[0].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.label
        
        if self.idx == (len(self.indices) - 1):
            self.idx = 0
            shuffle(self.indices)
        else:
            self.idx += 1


    def backward(self, top, propagate_down, bottom):
        pass


    def load_label(self, label_path):
        
        label = np.array(Image.open(label_path))
        if self.binary_mask:
            one_hot_label = np.zeros((1,192,256), dtype=np.float32)
            one_hot_label[0,label == 1] = 1.0
            one_hot_label[0,label == 2] = 1.0
            one_hot_label[0,label == 3] = 1.0
            #pdb.set_trace()
        else:
            one_hot_label = np.dstack([(label==1), (label==2), (label==3)]).astype(np.float32)
            one_hot_label = one_hot_label.transpose((2,0,1))
        return one_hot_label
        
