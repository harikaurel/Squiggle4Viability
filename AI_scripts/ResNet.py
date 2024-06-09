import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F

def conv3(in_channel, out_channel, stride=1, padding=1, groups=1):
  return nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, 
				   padding=padding, bias=False, dilation=padding, groups=groups)

def conv1(in_channel, out_channel, stride=1, padding=0):
  return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, 
				   padding=padding, bias=False)

def bcnorm(channel):
  return nn.BatchNorm1d(channel)

class Bottleneck(nn.Module):
	expansion = 1.5
	def __init__(self, in_channel, out_channel, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.bottelneck = nn.Sequential(
			conv1(in_channel, in_channel),
			bcnorm(in_channel),
			conv3(in_channel, in_channel, stride),
			bcnorm(in_channel),
			conv1(in_channel, out_channel),
			bcnorm(out_channel),
		)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
  
	def forward(self, x):
		identity = x

		out = self.bottelneck(x)
		
		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=2, chan_1=20, chan_2=30, chan_3=45, chan_4=67):
		super(ResNet, self).__init__()
		self.chan1 = 20

		# first block
		self.first_layer = nn.Sequential(
			nn.Conv1d(1, 20, 19, padding=5, stride=3),
			bcnorm(self.chan1),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(2, padding=1, stride=2)
		)

		self.avgpool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Linear(chan_4, 2)

		self.layer1 = self._make_layer(block, chan_1, layers[0])
		self.layer2 = self._make_layer(block, chan_2, layers[1], stride=2)
		self.layer3 = self._make_layer(block, chan_3, layers[2], stride=2)
		self.layer4 = self._make_layer(block, chan_4, layers[3], stride=2)
		#self.layer5 = self._make_layer(block, 100, layers[4], stride=2)

		# initialization
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm1d)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	
	def _make_layer(self, block, channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.chan1 != channels:
			downsample = nn.Sequential(
				conv1(self.chan1, channels, stride),
				bcnorm(channels),
			)

		layers = []
		layers.append(block(self.chan1, channels, stride, downsample))
		if stride != 1 or self.chan1 != channels:
			self.chan1 = channels
		for _ in range(1, blocks):
			layers.append(block(self.chan1, channels))

		return nn.Sequential(*layers)

	def _forward_impl(self, x, label=None, return_cam=False):
		x = x.unsqueeze(1)
		x = self.first_layer(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		pre_logits = self.avgpool(x)
		pre_logits = torch.flatten(pre_logits, 1)
		logits = self.fc(pre_logits)
		
		if return_cam:
			feature_map = x.detach().clone()
			cam_weights = self.fc.weight[label]
			cams = (cam_weights.view(*feature_map.shape[:2], 1) *
        		feature_map).mean(1, keepdim=False)
			return logits, cams

		return logits

	def forward(self, x, label=None, return_cam=False):
	  return self._forward_impl(x, label, return_cam)