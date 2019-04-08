import numpy as np
import cntk_unet
import cntk as C
import os
from cntk.logging import graph
from cntk import Trainer
from cntk.learners import learning_rate_schedule, UnitType
from cntk.learners import momentum_sgd, learning_parameter_schedule, momentum_schedule
from cntk.device import try_set_default_device, gpu
from cntk.losses import fmeasure,squared_error
from PIL import Image
from PIL import ImageOps
from cntk.train.training_session import CheckpointConfig, training_session
import cv2
try_set_default_device(gpu(0))

num_color_channels = 1
num_classes        = 3
image_width        = 256
image_height       = 256
epoch_size		   = 20
lr_per_mb = [0.2]*10 + [0.1]
momentum_per_mb = 0.9
l2_reg_weight = 0.00004
num_epochs = 20

Imagefile = os.listdir('data/image/')
Labelfile = os.listdir('data/label/')

minibatches_per_image = 10

def Img2CntkImg(path, resizeX, resizeY):
    img = Image.open(path)
    img = img.resize((resizeX, resizeY))  # width, height

    img = ImageOps.grayscale(img)

    training_img = np.array(img)
    training_img = np.array([training_img])
    training_img = training_img.astype(np.float32)
    return training_img

class MyDataSource(C.io.UserMinibatchSource):

	def __init__(self, f_dim, l_dim, minibatches_per_image, max_file):
		# Record the image dimensions for later
		self.f_dim, self.l_dim = f_dim, l_dim
		self.minibatches_per_image = minibatches_per_image
		self.num_color_channels, self.block_size, _ = self.f_dim
		self.already_loaded_images = False
		self.file_num = max_file
		self.current_mb_indices = 0

		# Record the stream information
		self.fsi = C.io.StreamInformation(
			'features', 0, 'dense', np.float32, self.f_dim)
		self.lsi = C.io.StreamInformation(
			'labels', 1, 'dense', np.float32, self.l_dim)
		self.images = []
		self.label_images = []
		for n in range(0,self.file_num):
			ansfile = "Data/Label/" + Labelfile[n]
			trnfile = "Data/Image/" + Imagefile[n]
			InputImage  = Img2CntkImg(trnfile,image_width,image_height)
			label_image = Img2CntkImg(ansfile,image_width,image_height)
			self.images.append(InputImage)
			self.label_images.append(label_image)
        

		super(MyDataSource, self).__init__()

	def stream_infos(self):
		return [self.fsi, self.lsi]

	def next_minibatch(self, mb_size_in_samples, number_of_workers=1, worker_rank=0,device=None):	
		features = np.zeros((mb_size_in_samples, self.num_color_channels,
							 self.block_size, self.block_size),
							dtype=np.float32)
		labels = np.zeros((mb_size_in_samples, self.num_color_channels,self.block_size,
						   self.block_size), dtype=np.float32)

		# Randomly select subsets of the image for training
		samples_retained = 0
		while samples_retained < mb_size_in_samples:
				features[samples_retained, :, :, :] = self.images[samples_retained]
				labels[samples_retained, :, :,:] = self.label_images[samples_retained]
				samples_retained += 1

		# Convert the label data to one-hot, then convert arrays to Values
		f_data = C.Value(batch=features)
		l_data = C.Value(batch=labels)

		result = {self.fsi: C.io.MinibatchData(
						f_data, mb_size_in_samples, mb_size_in_samples, False),
				  self.lsi: C.io.MinibatchData(
				  		l_data, mb_size_in_samples, mb_size_in_samples, False)}
		return(result)


def train():
	# Define the input variables
	f_dim = (1,image_width, image_height)
	l_dim = (1,image_width, image_height)

	feature = C.input_variable(f_dim, np.float32)
	label = C.input_variable(l_dim, np.float32)

	# Define the minibatch source
	minibatch_source = MyDataSource(f_dim, l_dim, minibatches_per_image, 30)
	input_map = {feature: minibatch_source.fsi,
					label: minibatch_source.lsi}

	z = cntk_unet.cntk_unet(feature)
	
	#loss = C.binary_cross_entropy(z,label)
	#loss = cntk_unet.dice_coefficient(z,label)
	loss  = C.fmeasure(z,label)
	progress_writers = [C.logging.progress_print.ProgressPrinter(
		tag='Training',
		num_epochs=num_epochs,
		freq=100)]

	lr_schedule = learning_parameter_schedule(lr_per_mb)
	mm_schedule = momentum_schedule(momentum_per_mb)
	learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
	trainer = Trainer(z, (loss,loss), learner, progress_writers)


	C.logging.progress_print.log_number_of_parameters(z)
	training_session(
		trainer=trainer,
		max_samples=2000,
		mb_source=minibatch_source, 
		mb_size=10,
		model_inputs_to_streams=input_map,
		checkpoint_config=CheckpointConfig(
			frequency=100,
			filename=os.path.join('trained_checkpoint.model'),
			preserve_all=True),
		progress_frequency=100
	).train()

	z.save('result.model')
train()






 