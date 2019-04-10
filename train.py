import numpy as np
import cntk_unet
import cntk as C
import os
import sys
from cntk.logging import graph , ProgressPrinter
from cntk import Trainer
from cntk.learners import learning_rate_schedule, UnitType
from cntk.learners import momentum_sgd, learning_parameter_schedule, momentum_schedule
from cntk.device import try_set_default_device, gpu
from cntk.losses import fmeasure,squared_error
from PIL import Image
from PIL import ImageOps
from cntk.train.training_session import CheckpointConfig, training_session
import cv2
import cntk.io.transforms as xforms
try_set_default_device(gpu(0))

num_color_channels = 1
num_classes        = 3
image_width        = 256
image_height       = 256
epoch_size		   = 100
lr_per_mb = [0.2]*10 + [0.1]
momentum_per_mb = 0.9
l2_reg_weight = 0.00001
num_epochs = 10
mb_size	= 2


Imagefile = os.listdir('data/image/')
Labelfile = os.listdir('data/label/')

minibatches_per_image = 5
maxsamplesize = num_epochs*100

def Img2CntkImg(path, resizeX, resizeY):
    img = Image.open(path)
    img = img.resize((resizeX, resizeY))  # width, height

    img = ImageOps.grayscale(img)

    training_img = np.array(img)
    training_img = np.array([training_img])
    training_img = training_img.astype(np.float32)
    return training_img

def create_reader(map_file1, map_file2):
    transforms = [xforms.scale(width=256, height=256, channels=1, interpolations='linear')]
    source1 = C.io.ImageDeserializer(map_file1, C.io.StreamDefs(
        source_image = C.io.StreamDef(field='image', transforms=transforms)))
    source2 = C.io.ImageDeserializer(map_file2, C.io.StreamDefs(
        target_image = C.io.StreamDef(field='image', transforms=transforms)))
    return C.io.MinibatchSource([source1, source2], max_samples=sys.maxsize, randomize=True)


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
			'labels', 0, 'dense', np.float32, self.l_dim)
		self.x = C.input_variable((self.block_size, self.block_size))
		self.oh_tf = C.one_hot(self.x, 1, False,
								  axis=0)
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

	def next_minibatch(self, mb_size_in_samples, numberf_of_workers=1, worker_rank=0,device=None):	
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
		l_data = C.Value(batch=self.oh_tf.eval({self.x: labels}))

		result = {self.fsi: C.io.MinibatchData(
						f_data, mb_size_in_samples, mb_size_in_samples, False),
				  self.lsi: C.io.MinibatchData(
				  		l_data, mb_size_in_samples, mb_size_in_samples, False)}
		return(result)



# Define the input variables
f_dim = (1,image_width, image_height)
l_dim = (1,image_width, image_height)

feature = C.input_variable(f_dim)
label = C.input_variable(l_dim)

# Define the minibatch source
reader = create_reader("image.txt", "label.txt")
#z = cntk_unet.cntk_unet(feature)
# Define the model
z = cntk_unet.model(1, 256,
                                2, [64, 128, 256, 512])(feature)

graph_description = C.logging.graph.plot(z, "graph.png")

#print(graph_description)

loss = C.fmeasure(z,label/255)
progress_printer = ProgressPrinter(tag='Training', num_epochs=num_epochs)
lr_schedule = learning_parameter_schedule(lr_per_mb)
mm_schedule = momentum_schedule(momentum_per_mb)
learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
trainer = Trainer(z, (loss, loss), learner, progress_printer)
input_map={
	feature: reader.streams.source_image,
	label: reader.streams.target_image
}

for epoch in range(num_epochs):       # loop over epochs
	sample_count = 0
	while sample_count < epoch_size:  # loop over minibatches in the epoch
		data = reader.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
		trainer.train_minibatch(data)                                    # update model with it
		sample_count += trainer.previous_minibatch_sample_count          # count samples processed so far
		if sample_count % (100 * mb_size) == 0:
			print ("Processed {0} samples".format(sample_count))

	trainer.summarize_training_progress()

z.save('result.model')





 