#!/usr/bin/env python
"""
Generate prototxt of the Spatial CNN for Caffe.
Paper: https://arxiv.org/pdf/1712.06080.pdf
"""
import argparse
import sys
import math

def generate_conv_layer(name, bottom, top, num_output, pad, kernel_size):
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Convolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    num_output: %d
    stride: %d
    pad: %d
    kernel_size: %d
  }
}\n'''%(name, bottom, top, num_output, stride, pad, kernel_size)
    return conv_layer_str

def generate_activation_layer(name, bottom, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "%s"
}\n'''%(name, bottom, top, act_type)
    return act_layer_str

def generate_fc_layer(name, bottom, top, num_output):
    fc_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "InnerProduct"
  param { 
    lr_mult: 1 
    decay_mult: 1 
    }
  
  param { 
    lr_mult: 2 
    decay_mult: 0 
    }
  inner_product_param {
    num_output: %d
    weight_filler {
      type: "gaussian"
      std: %d
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}\n'''%(name, bottom, top, num_output, std)
    return fc_layer_str

def generate_deconv_layer(name, bottom, top, num_output, pad, kernel_size):
    conv_layer_str = '''layer {
  name: "%s"
  bottom: "%s"
  top: "%s"
  type: "Deconvolution"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    weight_filler {
      type: "msra"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
    num_output: %d
    stride: %d
    pad: %d
    kernel_size: %d
  }
}\n'''%(name, bottom, top, num_output, stride, pad, kernel_size)
    return conv_layer_str

def main():
    
    network_str = ""
    network_str += generate_conv_layer('conv1','data','conv1',16,1,1,3)
    network_str += generate_activation_layer('relu1','conv1', 'conv1')
    network_str += generate_conv_layer('conv2','conv1','conv2',16,1,1,3)
    network_str += generate_activation_layer('relu2','conv2', 'conv2')
    network_str += generate_maxpool_layer('pool1','conv2','pool1')

    network_str += generate_conv_layer('conv3','pool1','conv3',32,1,1,3)
    network_str += generate_activation_layer('relu3','conv3', 'conv3')
    network_str += generate_conv_layer('conv4','conv3','conv4',32,1,1,3)
    network_str += generate_activation_layer('relu4','conv4', 'conv4')
    network_str += generate_maxpool_layer('pool2','conv4','pool2')

    network_str += generate_conv_layer('conv5','pool2','conv5',64,1,1,3)
    network_str += generate_activation_layer('relu5','conv5', 'conv5')
    network_str += generate_conv_layer('conv6','conv5','conv6',64,1,1,3)
    network_str += generate_activation_layer('relu6','conv6', 'conv6')
    network_str += generate_maxpool_layer('pool3','conv4','pool3')

    network_str += generate_conv_layer('conv7','pool3','conv7',128,1,1,3)
    network_str += generate_activation_layer('relu7','conv7', 'conv7')
    network_str += generate_conv_layer('conv8','conv7','conv8',128,1,1,3)
    network_str += generate_activation_layer('relu8','conv8', 'conv8')
    network_str += generate_maxpool_layer('pool4','conv8','pool4')

    network_str += generate_fc_layer('fc1', 'pool4', 'fc1', 64)
    network_str += generate_fc_layer('fc2', 'pool4', 'fc2', 24576)
    network_str += generate_reshape_layer('reshape', 'pool4', 'reshape', 12, 16, 128)

    
    network_str += generate_conv_layer('conv1','data','conv1',16,1,1,3)
    network_str += generate_activation_layer('relu1','conv1', 'conv1')
    network_str += generate_conv_layer('conv2','conv1','conv2',16,1,1,3)
    network_str += generate_activation_layer('relu2','conv2', 'conv2')
    network_str += generate_maxpool_layer('pool1','conv2','pool1')

    network_str += generate_conv_layer('conv3','pool1','conv3',32,1,1,3)
    network_str += generate_activation_layer('relu3','conv3', 'conv3')
    network_str += generate_conv_layer('conv4','conv3','conv4',32,1,1,3)
    network_str += generate_activation_layer('relu4','conv4', 'conv4')
    network_str += generate_maxpool_layer('pool2','conv4','pool2')

    network_str += generate_conv_layer('conv5','pool2','conv5',64,1,1,3)
    network_str += generate_activation_layer('relu5','conv5', 'conv5')
    network_str += generate_conv_layer('conv6','conv5','conv6',64,1,1,3)
    network_str += generate_activation_layer('relu6','conv6', 'conv6')
    network_str += generate_maxpool_layer('pool3','conv4','pool3')

    network_str += generate_conv_layer('conv7','pool3','conv7',128,1,1,3)
    network_str += generate_activation_layer('relu7','conv7', 'conv7')
    network_str += generate_conv_layer('conv8','conv7','conv8',128,1,1,3)
    network_str += generate_activation_layer('relu8','conv8', 'conv8')
    network_str += generate_maxpool_layer('pool4','conv8','pool4')


    fp = open(args.output, 'w')
    fp.write(scnn_pt)
    fp.close()

if __name__ == '__main__':
    main()
