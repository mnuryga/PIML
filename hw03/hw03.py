import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# constants
num_workers = 8
batch_size = 5000
num_epochs = 5
learning_rate = 0.02
progress_bar = True

# get device
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

class FFN_Model(nn.Module):
	'''
	simple feed forward network
	'''
	def __init__(self, in_dim, out_dim):
		super().__init__()
		# flatten images to 1d tensor
		self.flatten = nn.Flatten()

		# 2 fully connected layers
		self.fc1 = nn.Linear(in_dim, in_dim)
		self.fc2 = nn.Linear(in_dim, out_dim)

		# dropout layer
		self.dropout = nn.Dropout()

	def forward(self, x):
		'''
		forward pass function
		'''

		# flatten image
		x = self.flatten(x)

		# pass through both linear layers with dropout in between
		return self.fc2(self.dropout(self.fc1(x)))


# load datasets
train_dataset = torchvision.datasets.FashionMNIST("./data", download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST("./data", download = True, train = False, transform = transforms.Compose([transforms.ToTensor()]))

# print sizes of datasets
print('Before Split')
print(f'\t{len(train_dataset) = }')
print(f'\t{len(test_dataset) = }')

# no validation/dev set, so we create one from training data
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

# print sizes of datasets
print('After Split')
print(f'\t{len(train_dataset) = }')
print(f'\t{len(valid_dataset) = }')
print(f'\t{len(test_dataset) = }')

# initialize dataloaders
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size)

# print shape of batch
print(f'Batch Shape = {next(iter(train_loader))[0].shape}')

# display some images
fig, ax = plt.subplots(5,5)
for i in range(0, 5):
	for j in range(0, 5):
		ax[i][j].imshow(train_dataset[5*i+j][0].squeeze(), cmap='gray')
# plt.show()
# plt.clf()

# create model
model = FFN_Model(784, 10)
model = model.to(device)

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
	prev_loss.append(valid_loss)

	print(f'\tValidation loss = {valid_loss/(batch_idx+1)/batch_size:.6f}')

	# if valid_loss exceedes the 5-epoch rolling sum, break from training
	if valid_loss > np.mean(prev_loss[-5:]):
		break

num_correct = 0
sum_loss = 0
with torch.no_grad():
	for batch_idx, (images, labels) in enumerate(tqdm(test_loader, disable = not progress_bar, desc = 'Testing', ncols=60)):
		# send tensors to device
		images, labels = images.to(device), labels.to(device)

		# forward pass
		preds = model(images)

		# calculate loss
		loss = loss_func(preds, labels)
		sum_loss += loss.item()

		# calculate accuracy
		preds = torch.argmax(preds, axis=1)
		num_correct += torch.sum(preds == labels)

print(f'Testing:\n\tTest loss = {sum_loss/(batch_idx+1)/batch_size:.6f}')
print(f'\tTest Accuracy = {100*num_correct/len(test_dataset):.2f}%')

# Decision Tree
clf = LogisticRegression(max_iter = 40)

# get data
train_X = np.zeros((50000,784))
train_y = np.zeros((50000))
test_X = np.zeros((10000,784))
test_y = np.zeros((10000))

# process data into arrays
for i in range(len(train_dataset)):
	train_X[i] = train_dataset[i][0].flatten()
	train_y[i] = train_dataset[i][1]

for i in range(len(test_dataset)):
	test_X[i] = test_dataset[i][0].flatten()
	test_y[i] = test_dataset[i][1]

# train logistic classifier
clf.fit(train_X, train_y)

# get accuracy for logistic classifier
print(f'Logistic Regression Accuracy = {100*clf.score(test_X, test_y):.2f}%')