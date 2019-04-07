# Copyright (c) Microsoft. All rights reserved.
# Authors: Kolya Malkin, Nebojsa Jojic
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

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
    '''
    print ("upsampling")
    print(xr.shape)
    print(xx.shape)
    print(xy.shape)
    print(r.shape)
    '''
    return r

def cntk_unet(input): 
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
    conv10 = Convolution((3, 3), 1, init=glorot_uniform(), activation=sigmoid,pad= True)(conv9)      

    C.logging.graph.plot(conv10, 'model.png')  
    return conv10

def dice_coefficient(x, y):
    # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    #intersection = C.reduce_sum(x - y)
    err = squared_error(x,y,"se")
    return err #2 * intersection / (C.reduce_sum(x) + C.reduce_sum(y))

