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
    label_ = np.zeros((192,256))
    label_[label[0,:,:] == 1] = 1
    label_[label[1,:,:] == 1] = 2
    label_[label[2,:,:] == 1] = 3
    plt.imshow(label_)
    plt.show()

solver = caffe.SGDSolver('Models/solver.prototxt')
inference_net = caffe.Net('Models/inference_v2.prototxt', caffe.TEST)
for i in range(5):
    inference_net.forward()
    in_label = inference_net.blobs['data'].data
    out_label = inference_net.blobs['pred'].data
    plot_label(in_label)
    plot_label(out_label)
    solver.step(170)
    inference_net.share_with(solver.net)
    inference_net.forward()
    in_label = inference_net.blobs['data'].data
    out_label = inference_net.blobs['pred'].data
    plot_label(in_label)
    plot_label(out_label)
 
#net = caffe.Net('Models/train_prueba.prototxt', caffe.TEST)
#net.forward()
#in_label = net.blobs['data'].data
#mid_ = net.blobs['flattened'].data
#out_label = net.blobs['data_'].data
#plot_label(in_label)
#plot_label(out_label)
#pdb.set_trace()
#print "done"
