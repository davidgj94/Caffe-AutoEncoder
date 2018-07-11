import caffe
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil
import pickle
import time
from pathlib import Path
import parse
import pdb

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

def plot_label(label):
    label = np.squeeze(label)
    label_ = np.zeros((192,256))
    label_[label[1,:,:] == 1] = 1
    label_[label[2,:,:] == 1] = 2
    label_[label[3,:,:] == 1] = 3
    plt.imshow(label_)
    plt.show()

net = caffe.Net('Models/train_prueba.prototxt', caffe.TEST)
net.forward()
in_label = net.blobs['data'].data
mid_ = net.blobs['flattened'].data
out_label = net.blobs['data_'].data
plot_label(in_label)
plot_label(out_label)
print "done"
