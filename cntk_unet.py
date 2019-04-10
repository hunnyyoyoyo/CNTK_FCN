import numpy as np
import io
import cntk
from cntk.layers import *
from cntk.initializer import *
from cntk.ops import *
from cntk.cntk_py import squared_error
from cntk.layers import *


def UpSampling2D(x):
    xr = C.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2], 1))
    xx = C.splice(xr, xr, axis=-1)  # axis=-1 refers to the last axis
    xy = C.splice(xx, xx, axis=-3)  # axis=-3 refers to the middle axis
    r = C.reshape(xy, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2))
    return r

def cntk_unet(input,num_classes = 1): 
    conv1 = Convolution((3, 3), 64, init=glorot_uniform(), activation=relu,pad= True)(input)
    conv1 = Convolution((3, 3), 64, init=glorot_uniform(), activation=relu,pad= True)(conv1)
    pool1 = MaxPooling((2, 2), strides=(2, 2))(conv1)    

    conv2 = Convolution((3, 3), 128, init=glorot_uniform(), activation=relu,pad= True)(pool1)
    conv2 = Convolution((3, 3), 128, init=glorot_uniform(), activation=relu,pad= True)(conv2)
    pool2 = MaxPooling((2, 2), strides=(2, 2))(conv2)

    conv3 = Convolution((3, 3), 256, init=glorot_uniform(), activation=relu,pad= True)(pool2)
    conv3 = Convolution((3, 3), 256, init=glorot_uniform(), activation=relu,pad= True)(conv3)
    pool3 = MaxPooling((2, 2), strides=(2, 2))(conv3)

    conv4 = Convolution((3, 3), 512, init=glorot_uniform(), activation=relu,pad= True)(pool3)
    conv4 = Convolution((3, 3), 512, init=glorot_uniform(), activation=relu,pad= True)(conv4)
    drop4 = Dropout(0.5)(conv4)    
    pool4 = MaxPooling((2, 2), strides=(2, 2))(drop4)

    conv5 = Convolution((3, 3), 1024, init=glorot_uniform(), activation=relu,pad= True)(pool4)
    conv5 = Convolution((3, 3), 1024, init=glorot_uniform(), activation=relu,pad= True)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = C.splice(UpSampling2D(drop5), drop4, axis=0)
    conv6 = Convolution((3,3), 512, init=glorot_uniform(), activation=relu, pad=True)(up6)
    conv6 = Convolution((3,3), 512, init=glorot_uniform(), activation=relu, pad=True)(conv6)

    up7 = C.splice(UpSampling2D(conv6), conv3, axis=0)
    conv7 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(up7)
    conv7 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(conv7)

    up8 = C.splice(UpSampling2D(conv7), conv2, axis=0)
    conv8 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(up8)
    conv8 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(conv8)

    up9 = C.splice(UpSampling2D(conv8), conv1, axis=0)
    conv9 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(up9)
    conv9 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(conv9)
    #conv10 = Convolution((3, 3), 1, init=glorot_uniform(), activation=sigmoid,pad= True)(conv9) 
    # if num_classes > 1 then softmax is applied at loss function
    clf = Convolution2D((1, 1,), num_classes, activation=identity if num_classes > 1 else C.sigmoid, pad=True)(conv9)    

    #for node in clf.parameters:
     #   print(node.value) 

    return clf

def create_model(input, num_classes = 1):
    conv1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True)(input)
    conv1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True)(conv1)
    pool1 = MaxPooling((2,2), strides=(2,2))(conv1)

    conv2 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(pool1)
    conv2 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(conv2)
    pool2 = MaxPooling((2,2), strides=(2,2))(conv2)

    conv3 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(pool2)
    conv3 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(conv3)
    pool3 = MaxPooling((2,2), strides=(2,2))(conv3)

    conv4 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(pool3)
    conv4 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(conv4)
    pool4 = MaxPooling((2,2), strides=(2,2))(conv4)

    conv5 = Convolution((3,3), 512, init=glorot_uniform(), activation=relu, pad=True)(pool4)
    conv5 = Convolution((3,3), 512, init=glorot_uniform(), activation=relu, pad=True)(conv5)

    up6 = C.splice(UpSampling2D(conv5), conv4, axis=0)
    conv6 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(up6)
    conv6 = Convolution((3,3), 256, init=glorot_uniform(), activation=relu, pad=True)(conv6)

    up7 = C.splice(UpSampling2D(conv6), conv3, axis=0)
    conv7 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(up7)
    conv7 = Convolution((3,3), 128, init=glorot_uniform(), activation=relu, pad=True)(conv7)

    up8 = C.splice(UpSampling2D(conv7), conv2, axis=0)
    conv8 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(up8)
    conv8 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(conv8)

    up9 = C.splice(UpSampling2D(conv8), conv1, axis=0)
    conv9 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(up9)
    conv9 = Convolution((3,3), 64, init=glorot_uniform(), activation=relu, pad=True)(conv9)

    conv10 = Convolution((1,1), num_classes, init=glorot_uniform(), activation=sigmoid, pad=True)(conv9)

    return conv10

def conv_bn(input, filter_size, num_filters, strides=(1,1), init=uniform(0.00001)):
    c = Convolution(filter_size, num_filters, activation=None, init=init, pad=True, strides=strides, bias=False)(input)
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True)(c)
    return r

def conv_bn_relu(input, filter_size, num_filters, strides=(1,1), init=he_normal()):
    r = conv_bn(input, filter_size, num_filters, strides, init) 
    return relu(r)

def resnet_basic(input, num_filters):
    c1 = conv_bn_relu(input, (3,3), num_filters)
    c2 = conv_bn(c1, (3,3), num_filters)
    p  = c2 #+ input
    return relu(p)

def resnet_basic_inc(input, num_filters, strides=(2,2)):
    c1 = conv_bn_relu(input, (3,3), num_filters, strides)
    c2 = conv_bn(c1, (3,3), num_filters)
    c3  = conv_bn(input, (1,1), num_filters, strides)
    p  = c2#c3 + c2
    return relu(p)

def resnet_basic_stack(input, num_stack_layers, num_filters): 
    assert (num_stack_layers >= 0)
    l = input 
    for i in range(num_stack_layers): 
        l = resnet_basic(l, num_filters)
    return l 

def resu_model(input, num_stack_layers, c_map, num_classes, block_size):
    #r = cntk.slice(input, 0, 0, 1)
    #g = cntk.slice(input, 0, 1, 2)
    #b = cntk.slice(input, 0, 2, 3)
    #i = cntk.slice(input, 0, 3, 4)

    #r -= reduce_mean(r)
    #g -= reduce_mean(g)
    #b -= reduce_mean(b)
    #i -= reduce_mean(i)

    #input_do = splice(splice(splice(r, g, axis=0), b, axis=0), i, axis=0)

    conv = conv_bn(input, (3, 3), c_map[0])

    r1 = resnet_basic_stack(conv, num_stack_layers, c_map[0])

    r2_1 = resnet_basic_inc(r1, c_map[1])
    r2_2 = resnet_basic_stack(r2_1, num_stack_layers-1, c_map[1])
    
    r3_1 = resnet_basic_inc(r2_2, c_map[2])
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers-1, c_map[2])

    r4_1 = resnet_basic_inc(r3_2, c_map[3])
    r4_2 = resnet_basic_stack(r4_1, num_stack_layers-1, c_map[3])
  
    r4_us = layers.ConvolutionTranspose((3, 3), c_map[3], strides=2, output_shape=(block_size/4, block_size/4), pad=True, bias=False, init=bilinear(3, 3))(r4_2)

    o3 = relu(layers.Convolution((1, 1), c_map[2])(r3_2) + layers.Convolution((1, 1), c_map[2])(r4_us))
    o3_us = layers.ConvolutionTranspose((3, 3), c_map[2], strides=2, output_shape=(block_size/2, block_size/2), pad=True, bias=False, init=bilinear(3, 3))(o3)

    o2 = relu(layers.Convolution((1, 1), c_map[1])(r2_2) + layers.Convolution((1, 1), c_map[1])(o3_us))
    o2_us = layers.ConvolutionTranspose((3, 3), c_map[1], strides=2, output_shape=(block_size, block_size), pad=True, bias=False, init=bilinear(3, 3))(o2)

    o1 = relu(layers.Convolution((3, 3), c_map[0], pad=True)(input) + layers.Convolution((1, 1), c_map[0])(r1) + layers.Convolution((1, 1), c_map[0])(o2_us))

    return layers.Convolution((3, 3), num_classes, pad=True, activation=relu)(o1)

def model(c_classes, block_size, num_stack_layers, c_map):
    def tower(input):
        return resu_model(input, num_stack_layers, c_map, c_classes, block_size)
    return tower

