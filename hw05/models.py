import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class FFN_Model(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, p = 0.5):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(in_dim, hidden_dim*2)
		self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, out_dim)
		self.dropout1 = nn.Dropout(p = p)
		self.dropout2 = nn.Dropout(p = p)

	def forward(self, x):
		x = self.flatten(x)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc3(x)
		return x

class Conv_Model(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 6, 4, padding = 2)
		self.maxpool1 = nn.MaxPool2d(2, stride = 2)
		self.conv2 = nn.Conv2d(6, 16, 4)
		self.maxpool2 = nn.MaxPool2d(2, stride = 2)
		self.ffn = FFN_Model(in_dim, hidden_dim, out_dim)

	def forward(self, x):
		x = self.conv1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.maxpool2(x)
		x = self.ffn(x)
		return x

class MobileNet_Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.pre = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
		for p in self.pre.parameters():
			p.requires_grad = False
		self.ffn = FFN_Model(1000, 32, 10)

	def forward(self, x):
		x = self.pre(x)
		return self.ffn(x)

class VAE(nn.Module):
	'''
	Variational Autoencoder as described in https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
	'''
	def __init__(self, in_channels, feature_dim = 32*20*20, hidden_dim = 256):
		super(VAE, self).__init__()
		# encoder
		self.enc_conv1 = nn.Conv2d(in_channels, 16, 5)
		self.enc_conv2 = nn.Conv2d(16, 32, 5)
		self.enc_fc1 = nn.Linear(feature_dim, hidden_dim)
		self.enc_fc2 = nn.Linear(feature_dim, hidden_dim)

		# decoder
		self.dec_fc1 = nn.Linear(hidden_dim, feature_dim)
		self.dec_conv1 = nn.ConvTranspose2d(32, 16, 5)
		self.dec_conv2 = nn.ConvTranspose2d(16, in_channels, 5)

	def encoder(self, x):
		x = F.relu(self.enc_conv1(x))
		x = F.relu(self.enc_conv2(x))
		x = x.view(-1, 32*20*20)
		mu = self.enc_fc1(x)
		logVar = self.enc_fc2(x)
		return mu, logVar

	def reparameterize(self, mu, logVar):
		std = torch.exp(logVar/2)
		eps = torch.randn_like(std)
		return mu + std * eps

	def decoder(self, z):
		x = F.relu(self.dec_fc1(z))
		x = x.view(-1, 32, 20, 20)
		x = F.relu(self.dec_conv1(x))
		x = torch.sigmoid(self.dec_conv2(x))
		return x

	def forward(self, x):
		mu, logVar = self.encoder(x)
		z = self.reparameterize(mu, logVar)
		out = self.decoder(z)
		return out, mu, logVar