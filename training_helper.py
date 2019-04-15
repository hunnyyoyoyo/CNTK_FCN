import numpy as np
import cntk_resnet_fcn
import cntk as C
from cntk.learners import learning_rate_schedule, UnitType

#-------------------------------------------------------------------------------
def slice_minibatch(data_x, data_y, i, minibatch_size):
    sx = data_x[i * minibatch_size:(i + 1) * minibatch_size]
    sy = data_y[i * minibatch_size:(i + 1) * minibatch_size]
    
    return sx, sy

#-------------------------------------------------------------------------------
def measure_error(source, data_x_files, data_y_files, x, y, trainer, minibatch_size):
    errors = []
    for i in range(0, int(len(data_x_files) / minibatch_size)):
        data_sx_files, data_sy_files = slice_minibatch(data_x_files, data_y_files, i, minibatch_size)
        data_sx, data_sy = source.files_to_data(data_sx_files, data_sy_files)
        errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))

    return np.mean(errors)
#-------------------------------------------------------------------------------