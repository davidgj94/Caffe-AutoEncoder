import caffe; caffe.set_mode_gpu()
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
    label = np.around(label)
    label_ = np.zeros((192,256,3))
    label_[label[0,:,:] == 1,0] = 255
    label_[label[1,:,:] == 1,1] = 255
    label_[label[2,:,:] == 1,2] = 255
    plt.imshow(label_)
    plt.show()

def plot_label_binary(label):
    label = np.squeeze(label)
    label = np.around(label)
    plt.imshow(label)
    plt.show()
    
solver = caffe.SGDSolver('Models/solver.prototxt')
inference_net = caffe.Net('Models/inference_reduced.prototxt', caffe.TEST)
for i in range(5):
    solver.step(170)
    inference_net.share_with(solver.net)
    inference_net.forward()
    in_label = inference_net.blobs['data'].data
    out_label = inference_net.blobs['pred'].data
    plot_label_binary(in_label)
    plot_label_binary(out_label)
 
#net = caffe.Net('Models/train_prueba.prototxt', caffe.TEST)
#net.forward()
#in_label = net.blobs['data'].data
#mid_ = net.blobs['flattened'].data
#out_label = net.blobs['data_'].data
#plot_label_binary(in_label)
#plot_label_binary(out_label)
#pdb.set_trace()
#print "done"
