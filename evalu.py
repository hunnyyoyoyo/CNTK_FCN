import numpy as np
import cntk as C
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from cntk import load_model
from PIL import Image
from PIL import ImageOps


num_color_channels = 1
image_width        = 256
image_height       = 256

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)   

def Img2CntkImg(path, resizeX, resizeY):
    img = Image.open(path)
    img = img.resize((resizeX, resizeY))  # width, height

    img = ImageOps.grayscale(img)

    training_img = np.array(img)
    training_img = np.array([training_img])
    training_img = training_img.astype(np.float32)
    return training_img

testimage = Img2CntkImg('test.png',image_width,image_height)

model = load_model('result.model')
# compute model output
arguments = {model.arguments[0]: [testimage]}
out_put = model.eval(arguments)
print(out_put.shape)






