import numpy as np
import cntk as C
import cv2
import os
import time
import coco   # local class to read the COCO images and labels
import helper # some functions to plot images
import gc
from tqdm import tqdm

data_path = os.path.join("data/M4")
zip_path = os.path.join("data-zip")
model_path = os.path.join("Models")

print ("Using Microsoft Cognitive Toolkit version {}".format(C.__version__))
print ("Using numpy version {}".format(np.__version__))

make_model = False

print('[i] Configuring data source...')
try:
    source = coco.CocoMs(os.path.join(data_path, "CocoMS"))
    validation_input_image_files, validation_target_mask_files = source.get_data(train_data_folder='/Validation')
    print('[i] # validation samples: ', len(validation_input_image_files))
    print('[i] # classes:            ', source.num_classes)
    print('[i] Image size:           ', (224,224))
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))


# Drawing

def process_images():
    print("[i] Started image processing...", flush=True)
    tic = time.time()
    
    input_images_rgb = []
    for x in tqdm(validation_input_images, ascii=True, desc='[i] Converting input images (BGR2RGB)...'):
        img = np.moveaxis(x,0,2).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_images_rgb.append(img)

    target_masks_rgb=[]
    for x in tqdm(validation_target_masks, ascii=True, desc='[i] Coloring ground truth images...'):
        target_masks_rgb.append(helper.masks_to_colorimg(x))

    pred_rgb=[]
    for x in tqdm(pred, ascii=True, desc='[i] Coloring prediction images...'):
        pred_rgb.append(helper.masks_to_colorimg(x))

    output_images_rgb = []
    for index in tqdm(range(len(input_images_rgb)), ascii=True, desc='[i] Combining input images + predictions...'):
        img = cv2.bitwise_or(input_images_rgb[index], pred_rgb[index])
        output_images_rgb.append(img)

    print('Image Processing time: {} s.'.format(time.time() - tic))
    print("Image processing finished... Now plotting (this can take a while - 2-3 minutes on Azure Notebooks)...", flush=True)
    helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb, output_images_rgb])

#
# We need to convert our validation filenames to image data for the predictor...
#

print ("[i] Converting file lists to image data...", flush=True)
tic = time.time()
validation_input_images, validation_target_masks = \
        source.files_to_data(validation_input_image_files, validation_target_mask_files)
print('Converting validation image data time: {} s.'.format(time.time() - tic))
print("[i] Converting file lists finished...")

model = C.load_model('trained.model')

# Prediction

print("[i] Starting prediction...", flush=True)

pred = []
for idx in tqdm(range(0, len(validation_input_images)),ascii=True, desc='[i] Predicting...'):
    pred += list(model.eval(validation_input_images[idx]))

print('[i] {} images predicted.'.format(len(pred)))
print("[i] Prediction finished...")

process_images()
print("Garbage collection reclaimed {} objects".format(gc.collect()))