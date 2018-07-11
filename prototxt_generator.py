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
}\n'''%(name, bottom, bottom, act_type)
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
    args = parse_args()
    scnn_pt = generate_SCNN(args)

    fp = open(args.output, 'w')
    fp.write(scnn_pt)
    fp.close()

if __name__ == '__main__':
    main()
