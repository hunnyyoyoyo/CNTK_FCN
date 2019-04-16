# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import cntk as C
import numpy as np
import time
import os
import cv2
import coco   # local class to read the COCO images and labels
import training_helper  #
import cntk_resnet_fcn
from cntk.learners import learning_rate_schedule, UnitType
from cntk.device import try_set_default_device, gpu
from tqdm import tqdm
from cntk.logging import ProgressPrinter
from cntk_resnet_unet import create_transfer_learning_model
from cntk.losses import fmeasure

# Paths relative to current python file.
data_path = os.path.join("data/M4")
zip_path = os.path.join("data-zip")
model_path = os.path.join("Models")
base_model_file = os.path.join(model_path, "ResNet50_ImageNet_CNTK.model")
make_model = False

def train(train_image_files, train_mask_files, val_image_files, val_mask_files, base_model_file, freeze=False):
    # Create model
    sample_img, sample_mask = source.files_to_data([train_image_files[0]], [val_image_files[0]])
    x = C.input_variable(sample_img[0].shape)
    y = C.input_variable(sample_mask[0].shape)
    
    z = create_transfer_learning_model(x, source.num_classes, base_model_file, freeze)
    dice_coef = fmeasure(z, y)


    mean_ce, pe = training_helper.criteria(y, z/255, 224,
						                    source.num_classes, [0.0, 1.0])


    # Prepare model and trainer
    if (isUsingGPU):
        lr_mb = [0.001] * 5 + [0.0001] * 5 + [0.00001]*5 + [0.000001]*5 + [0.0000001]*5
    else:
        # training without a CPU is really slow, so we'll deliberatly shrink the amount of training
        # to just an epoch if we're on a CPU - just to give a flavor of what happens during training
        # and then read in a pre-trained model for inference instead.
        lr_mb = [0.0001] * 1 # deliberately shrink if training on CPU...

    # Get minibatches of training data and perform model training
    minibatch_size = 2
    num_epochs = len(lr_mb)
        
    progress_printer = ProgressPrinter(tag='Training', num_epochs=num_epochs)
    lr = learning_rate_schedule(lr_mb, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0.9)
    trainer = C.Trainer(z, (mean_ce, pe), C.learners.adam(z.parameters, lr=lr, momentum=momentum),progress_printer)
    
    training_errors = []
    test_errors = []
   
    for e in range(0, num_epochs):
        for i in tqdm(range(0, int(len(train_image_files) / minibatch_size)), ascii=True, 
                                desc="[i] Processing epoch {}/{}".format(e, num_epochs-1)):
            data_x_files, data_y_files = training_helper.slice_minibatch(train_image_files, train_mask_files, i, minibatch_size)
            data_x, data_y = source.files_to_data(data_x_files, data_y_files)
            trainer.train_minibatch({z.arguments[0]: data_x, y: data_y})
        
        trainer.summarize_training_progress()
        # Measure training error
        training_error = training_helper.measure_error(source, data_x_files, data_y_files, z.arguments[0], y, trainer, minibatch_size)
        training_errors.append(training_error)
        
        # Measure test error
        test_error = training_helper.measure_error(source, val_image_files, val_mask_files, z.arguments[0], y, trainer, minibatch_size)
        test_errors.append(test_error)

    z.save('trained.model')
    print("epoch #{}: training_error={}, test_error={}".format(e, training_errors[-1], test_errors[-1]))    

print ("Using Microsoft Cognitive Toolkit version {}".format(C.__version__))
print ("Using numpy version {}".format(np.__version__))

try:
    isUsingGPU = C.device.try_set_default_device(C.device.gpu(0))
except ValueError:
    isUsingGPU = False
    C.device.try_set_default_device(C.device.cpu())
    
print ("[i] The Cognitive Toolkit is using the {} for processing".format("GPU" if isUsingGPU else "CPU"))

# Configure the data source
print('[i] Configuring data source...')
try:
    source = coco.CocoMs(os.path.join(data_path, "CocoMS"))
    training_input_image_files, training_target_mask_files = source.get_data(train_data_folder='/Training')
    validation_input_image_files, validation_target_mask_files = source.get_data(train_data_folder='/Validation')
    print('[i] # training samples:   ', len(training_input_image_files))
    print('[i] # validation samples: ', len(validation_input_image_files))
    print('[i] # classes:            ', source.num_classes)
    print('[i] Image size:           ', (224,224))

    train(training_input_image_files,training_target_mask_files,validation_input_image_files,validation_target_mask_files,base_model_file)
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))   


