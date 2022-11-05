import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
from models import Conv_Model, MobileNet_Model

# constants
num_workers = 4
batch_size = 128
num_epochs = 15
learning_rate = 0.001
progress_bar = True


# get device
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")
# EDA
# create dataset

def get_loaders(transforms):
	train_dataset = torchvision.datasets.CIFAR10('../datasets/', train=True, download=True, transform=transforms)
	_dataset = torchvision.datasets.CIFAR10('../datasets/', train=False, download=True, transform=transforms)

	# split nontraining dataset into validation and testing datasets
	nontrain_length = len(_dataset)//2
	valid_dataset, test_dataset = torch.utils.data.random_split(_dataset, [nontrain_length, nontrain_length], generator=torch.Generator().manual_seed(0))

	# create dataloaders
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, num_workers = 2, shuffle = False, persistent_workers = True)
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, num_workers = 2, shuffle = False, persistent_workers = True)
	test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, num_workers = 1, shuffle = False, persistent_workers = True)

	return train_loader, valid_loader, test_loader

def train(model, train_loader, valid_loader):
	# initialize optimizer
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)

	# initialize loss function
	loss_func = nn.CrossEntropyLoss()

	# prev_loss is used to store validation losses -> training is stopped
	# once validation loss is above a 5-epoch rolling mean
	prev_loss = []

	# iterate for specified number of epochs
	for epoch in range(num_epochs):
		model.train()
		sum_loss = 0
		for batch_idx, (images, labels) in enumerate(tqdm(train_loader, disable = not progress_bar, desc = f'Epoch {epoch:02d}', ncols=60)):
			# send tensors to device
			images, labels = images.to(device), labels.to(device)

			# zero out gradients
			optimizer.zero_grad()

			# forward pass
			preds = model(images)

			# calculate loss
			loss = loss_func(preds, labels)
			sum_loss += loss.item()

			# backward pass
			loss.backward()

			# step optimizer
			optimizer.step()

		print(f'\tTrain loss =      {sum_loss/(batch_idx+1)/batch_size:.6f}')

		# validation loop
		model.eval()
		valid_loss = 0
		with torch.no_grad():
			for batch_idx, (images, labels) in enumerate(valid_loader):
				# send tensors to device
				images, labels = images.to(device), labels.to(device)

				# forward pass
				preds = model(images)

				# calculate loss
				loss = loss_func(preds, labels)
				valid_loss += loss.item()

		# append current loss to prev_loss list
		prev_loss.append(valid_loss/(batch_idx+1)/batch_size)

		print(f'\tValidation loss = {valid_loss/(batch_idx+1)/batch_size:.6f}')

		# # if valid_loss exceedes the 5-epoch rolling sum, break from training
		if valid_loss/(batch_idx+1)/batch_size > np.mean(prev_loss[-5:]):
			continue
			break

	return model, prev_loss

def test(model, test_loader):
	loss_func = nn.CrossEntropyLoss()

	sum_loss = 0
	num_correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for batch_idx, (images, labels) in enumerate(tqdm(test_loader, disable = not progress_bar, desc = 'Testing', ncols=60)):
			# send tensors to device
			images, labels = images.to(device), labels.to(device)

			# forward pass
			preds = model(images)

			# calculate loss
			# loss = loss_func(preds.float(), F.one_hot(labels, num_classes = 10).float())
			loss = loss_func(preds, labels)
			sum_loss += loss.item()

			# calc number correct
			preds = torch.argmax(preds, dim = 1)
			num_correct += torch.sum(preds == labels)
			total += len(preds)

	print(f'Test loss: {sum_loss/(batch_idx+1)/batch_size:.6f}')
	print(f'Test acc:  {num_correct/total:.6f}')
	return sum_loss

# train and test each model
print('--------Conv---------')
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
convert = torchvision.transforms.ConvertImageDtype(torch.float)
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),normalize, convert])
train_loader, valid_loader, test_loader = get_loaders(transforms)
Conv_model = Conv_Model(6*6*16, 128, 10)
Conv_model.to(device)

Conv_model, Conv_losses = train(Conv_model, train_loader, valid_loader)
Conv_loss = test(Conv_model, test_loader)


print('------MobileNet------')
train_loader, valid_loader, test_loader = get_loaders(MobileNet_V2_Weights.IMAGENET1K_V1.transforms())
MobileNet_model = MobileNet_Model()

MobileNet_model.to(device)

MobileNet_model, MobileNet_losses = train(MobileNet_model, train_loader, valid_loader)
MobileNet_loss = test(MobileNet_model, test_loader)

# affine transformations -> -45 to +45 degree rotationsm -20% to +20% translations in x and y directions, and 90% to 110% scale factor
affine = torchvision.transforms.RandomAffine(45, translate = (0.2, 0.2), scale = (0.9, 1.1))

# same transformations from earlier
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.225, 0.225, 0.225])
convert = torchvision.transforms.ConvertImageDtype(torch.float)
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, convert, affine])
train_loader, valid_loader, test_loader = get_loaders(transforms)

# create model and send to gpu
Conv_model = Conv_Model(6*6*16, 128, 10)
Conv_model.to(device)

Conv_model, Conv_losses = train(Conv_model, train_loader, valid_loader)
Conv_loss = test(Conv_model, test_loader)

'''
print('--------LSTM--------')
LSTM_model, LSTM_losses = train(LSTM_model, train_loader, valid_loader)
LSTM_loss = test(LSTM_model, test_loader)

print('--------GRU---------')
GRU_model, GRU_losses = train(GRU_model, train_loader, valid_loader)
GRU_loss = test(GRU_model, test_loader)

# plot losses over time
xx = np.arange(num_epochs)
plt.plot(xx, RNN_losses, label = 'RNN')
plt.plot(xx, LSTM_losses, label = 'LSTM')
plt.plot(xx, GRU_losses, label = 'GRU')
plt.legend()
plt.title('Validation Loss')
plt.show()

# store embeddings as dict
embeddings = {}
with open('glove.6B.50d.txt', 'r', encoding='cp437') as f:
	for row in f:
		vals = row.strip().split(' ')
		# the first value is the word, all the rest are embeddings
		embeddings[vals[0]] = np.array(vals[1:]).astype(np.float32)

def cosine_similarity(embeddings, w1, w2):
	# get embeddings for each word
	v1 = embeddings[w1]
	v2 = embeddings[w2]

	# calculate cosine similarity
	return (v1.dot(v2))/(np.linalg.norm(v1) * np.linalg.norm(v2))

def dissimilarity_metric(embeddings, w1, w2):
	# get embeddings for each word
	v1 = embeddings[w1]
	v2 = embeddings[w2]

	# calculate euclidean distance
	return np.linalg.norm(v1 - v2)
'''