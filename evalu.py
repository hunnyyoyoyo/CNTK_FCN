import numpy as np
import cntk as C
from cntk import load_model
from PIL import Image
from PIL import ImageOps

num_color_channels = 1
image_width        = 256
image_height       = 256

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





