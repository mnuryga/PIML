import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# constants
num_workers = 8
batch_size = 1024
num_epochs = 20
learning_rate = 0.001
window = 64
progress_bar = True

# get device
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

class Temp_Dataset(Dataset):
	def __init__(self, temps, window):
		super().__init__()
		self.temps = temps
		self.window = window

	def __getitem__(self, x):
		# return time series and next value
		return self.temps[x:x + self.window], self.temps[x + self.window]

	def __len__(self):
		return len(self.temps) - self.window


class Model(nn.Module):
	def __init__(self, window, hidden_dim, out_dim, nn_type):
		super().__init__()

		# flatten function
		self.flatten = nn.Flatten()

		# select specified RNN
		if nn_type == 'RNN':
			self.rnn = nn.RNN(input_size = window, hidden_size = hidden_dim, num_layers = 2, batch_first = True, dropout = 0.5)
		elif nn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size = window, hidden_size = hidden_dim, num_layers = 2, batch_first = True, dropout = 0.5)
		elif nn_type == 'GRU':
			self.rnn = nn.GRU(input_size = window, hidden_size = hidden_dim, num_layers = 2, batch_first = True, dropout = 0.5)

		self.fc1 = nn.Linear(hidden_dim, hidden_dim)
		self.drop = nn.Dropout(0.5)
		self.fc2 = nn.Linear(hidden_dim, out_dim)

	def forward(self, x):
		out, (hn, cn) = self.rnn(x)
		x = self.flatten(out)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.drop(x)
		x = self.fc2(x)
		return x

def main():
	# EDA

	# read in data
	temps = pd.read_csv('temperature.csv', parse_dates=['datetime'])
	temps = temps.set_index('datetime')

	# print shape
	print(f'Dataset shape: {temps.shape}')

	# extract only one city's worth of temp data
	temps = temps[['Minneapolis']]

	# print stats on dataset
	print(temps.describe())

	# temps are in Kelvin, we can convert to Celsius
	temps = temps - 273.15

	# plot temp over time
	temps.plot()
	# plt.show()
	plt.clf()

	# plot temp over time in a smaller window
	temps[100:200].plot()
	# plt.show()
	plt.clf()

	# extract temps into np array
	temps = temps.to_numpy()[1:].T[0].astype(np.float32)
	# remove any nans
	temps = temps[~np.isnan(temps)]

	# check shape of temps
	print(f'{temps.shape = }')

	# split into train/validation/test
	n = len(temps)
	train_temps = temps[:int(0.8*n)]
	valid_temps = temps[int(0.8*n):int(0.9*n)]
	test_temps = temps[int(0.9*n):]

	# create datasets
	train_dataset = Temp_Dataset(train_temps, window)
	valid_dataset = Temp_Dataset(valid_temps, window)
	test_dataset = Temp_Dataset(test_temps, window)

	# create dataloaders
	train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
	valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size)
	test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size)

	# initalize models
	RNN_model = Model(window, 32, 1, 'RNN')
	RNN_model = RNN_model.to(device)

	LSTM_model = Model(window, 32, 1, 'LSTM')
	LSTM_model = LSTM_model.to(device)

	GRU_model = Model(window, 32, 1, 'GRU')
	GRU_model = GRU_model.to(device)


	def train(model, train_loader, valid_loader):
		# initialize optimizer
		optimizer = optim.Adam(model.parameters(), lr = learning_rate)

		# initialize loss function
		loss_func = nn.MSELoss()

		# prev_loss is used to store validation losses -> training is stopped
		# once validation loss is above a 5-epoch rolling mean
		prev_loss = []

		# iterate for specified number of epochs
		for epoch in range(num_epochs):
			model.train()
			sum_loss = 0
			for batch_idx, (temps, labels) in enumerate(tqdm(train_loader, disable = not progress_bar, desc = f'Epoch {epoch:02d}', ncols=60)):
				# send tensors to device
				temps, labels = temps.to(device), labels.to(device)

				# zero out gradients
				optimizer.zero_grad()

				# forward pass
				preds = model(temps.float()).T[0]

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
				for batch_idx, (temps, labels) in enumerate(valid_loader):
					# send tensors to device
					temps, labels = temps.to(device), labels.to(device)

					# forward pass
					preds = model(temps.float()).T[0]

					# calculate loss
					loss = loss_func(preds, labels)
					valid_loss += loss.item()

			# append current loss to prev_loss list
			prev_loss.append(valid_loss/(batch_idx+1)/batch_size)

			print(f'\tValidation loss = {valid_loss/(batch_idx+1)/batch_size:.6f}')

			# # if valid_loss exceedes the 5-epoch rolling sum, break from training
			# if valid_loss/(batch_idx+1)/batch_size > np.mean(prev_loss[-5:]):
			# 	break

		return model, prev_loss

	def test(model, test_loader):
		loss_func = nn.MSELoss()

		sum_loss = 0
		model.eval()
		with torch.no_grad():
			for batch_idx, (temps, labels) in enumerate(tqdm(test_loader, disable = not progress_bar, desc = 'Testing', ncols=60)):
				# send tensors to device
				temps, labels = temps.to(device), labels.to(device)

				# forward pass
				preds = model(temps.float()).T[0]

				# calculate loss
				loss = loss_func(preds, labels)
				sum_loss += loss.item()

		print(f'Test loss: {sum_loss/(batch_idx+1)/batch_size:.6f}')
		return sum_loss

	# train and test each model
	print('--------RNN---------')
	RNN_model, RNN_losses = train(RNN_model, train_loader, valid_loader)
	RNN_loss = test(RNN_model, test_loader)

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

if __name__ == '__main__':
	main()