import argparse
import numpy
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


class ConvNet(object):
	"""ConvNet Model class"""
	def __init__(self,
				 arch="resnet18",
				 ckp="/model/model_best.pth.tar",
				 train_dir="/input/train",
				 evalf="/eval"):
		"""MNIST ConvNet Builder
		Args:
			ckp: path to model checkpoint file (to continue training).
			evalf: path to evaluate sample.
		"""
		# Path to model weight
		self._ckp = ckp
		# Use CUDA?
		self._cuda = torch.cuda.is_available()
		try:
			os.path.isfile(ckp)
			self._ckp = ckp
		except IOError as e:
			# Does not exist OR no read permissions
			print ("Unable to open ckp file")
		self._evalf = evalf
		self._arch = arch
		 # Size on model
		if arch.startswith('inception'):
			self._size = (299, 299)
		else:
			self._size = (224, 256)
		# Get labels
		self._labels = self._get_label(train_dir)


	# Build the model loading the weights
	def build_model(self):
		# Create model from scratch or use a pretrained one
		print("=> using model '{}'".format(self._arch))
		self._model = models.__dict__[self._arch](num_classes=len(self._labels))
		print("=> loading checkpoint '{}'".format(self._ckp))
		if self._cuda:
			checkpoint = torch.load(self._ckp)
		else:
			# Load GPU model on CPU
			checkpoint = torch.load(self._ckp, map_location=lambda storage, loc: storage)
		# Load weights
		self._model.load_state_dict(checkpoint['state_dict'])

		if self._cuda:
			self._model.cuda()
		else:
			self._model.cpu()


	# Preprocess Images to be ImageNet-compliant
	def image_preprocessing(self):
		"""Take images from args.evalf, process to be ImageNet compliant
		and classify them with ImageNet ConvNet model chosen"""
		# Normalize on RGB Value
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		def pil_loader(path):
			"""Load images from /eval/ subfolder and resized it as squared"""
			with open(path, 'rb') as f:
				with Image.open(f) as img:
					sqrWidth = numpy.ceil(numpy.sqrt(img.size[0]*img.size[1])).astype(int)
					return img.resize((sqrWidth, sqrWidth))

		self._test_loader = torch.utils.data.DataLoader(
			datasets.ImageFolder(self._evalf, transforms.Compose([
				transforms.Scale(self._size[1]), # 256
				transforms.CenterCrop(self._size[0]), # 224 , 299
				transforms.ToTensor(),
				normalize,
			]), loader=pil_loader),
			batch_size=1, shuffle=False,
			num_workers=1, pin_memory=False)


	def _get_label(self, train_dir):
		# Normalize on RGB Value
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		# Train -> Preprocessing -> Tensor
		train_dataset = datasets.ImageFolder(
			train_dir,
			transforms.Compose([
				transforms.RandomSizedCrop(self._size[0]), #224 , 299
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]))

		# Get number of labels
		return train_dataset.classes


	def classify(self):
		"""Classify the current test batch"""
		self._model.eval()
		for data, _ in self._test_loader:
			if self._cuda:
				data = data.cuda()
			data = torch.autograd.Variable(data, volatile=True)
			output = self._model(data)
			# Take last layer output
			if isinstance(output, tuple):
				output = output[len(output)-1]

			lab = self._labels[numpy.asscalar(output.data.max(1, keepdim=True)[1].cpu().numpy())]
			print (self._labels, lab)
		return lab