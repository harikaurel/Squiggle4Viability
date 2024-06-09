import os
import click
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from dataset import SingleFileDataset
from ResNet import ResNet
from ResNet import Bottleneck
from ResNet4Sequence import ResNet4Sequence
from ResNet4Sequence import Bottleneck4Sequence
from Transformer import Transformer
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR
import torch.optim as optim

@click.command()
@click.option('--tTrain', '-tt', help='The path of target sequence training set', type=click.Path(exists=True))
@click.option('--tVal', '-tv', help='The path of target sequence validation set', type=click.Path(exists=True))
@click.option('--nTrain', '-nt', help='The path of non-target sequence training set', type=click.Path(exists=True))
@click.option('--nVal', '-nv', help='The path of non-target sequence validation set', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='The output path for trained model')
@click.option('--interm', '-i', help='The path and name for model checkpoint (optional)', 
																type=click.Path(exists=True), required=False)
@click.option('--batch', '-b', default=1000, help='Batch size, default 1000')
@click.option('--epoch', '-e', default=100, help='Number of epoches, default 100')
@click.option('--learningrate', '-l', default=1e-4, help='Learning rate, default 1e-4')
@click.option('--model_type', '-m', default='ResNet1', 
					help='Model to use for training, options are "ResNet1", "ResNet2", "ResNet3", "Transformer" or "ResNet4Sequence"')
@click.option('--sig_len', '-sl', default=10000, help='signal length')
@click.option('--num_letter', "-nl", default=4, help='number of letters (A, C, G, T)')

def main(ttrain, tval, ntrain, nval, outpath, interm, batch, epoch, learningrate, model_type, sig_len, num_letter):

	click.echo(f"""
	tTrain: {ttrain}
	tVal: {tval}
	nTrain: {ntrain}
	nVal: {nval}
	outpath: {outpath}
	interm: {interm}
	batch: {batch}
	epoch: {epoch}
	learningrate: {learningrate}
	model_type: {model_type}
	sig_len: {sig_len}
	num_letter: {num_letter}
	""")
 
	if torch.cuda.is_available():
		device = torch.device('cuda')
		print(device)
		print(torch.cuda.device_count())
	else:
		device = torch.device('cpu')
		print(device)

	params = {'batch_size': batch,
				'shuffle': True, 
				'num_workers': 12}


	pos_train_dataset = SingleFileDataset(ttrain, 1)  # 1 for dead
	neg_train_dataset = SingleFileDataset(ntrain, 0)  # 0 for alive
	training_set = ConcatDataset([pos_train_dataset, neg_train_dataset])
	training_generator = DataLoader(training_set, **params)

	pos_val_dataset = SingleFileDataset(tval, 1)
	neg_val_dataset = SingleFileDataset(nval, 0)
	validation_set = ConcatDataset([pos_val_dataset, neg_val_dataset])
	validation_generator = DataLoader(validation_set, **params)
 
	print("Number of samples in pos_train_dataset:", len(pos_train_dataset))
	print("Number of samples in neg_train_dataset:", len(neg_train_dataset))
	print("Number of samples in pos_val_dataset:", len(pos_val_dataset))
	print("Number of samples in neg_val_dataset:", len(neg_val_dataset))
	
	os.makedirs(outpath, exist_ok=True)
		
	### load model
	if model_type == 'ResNet1':
		bmodel = ResNet(Bottleneck, [2, 2, 2, 2])
		print("model", bmodel)
	elif model_type == 'ResNet2':
		bmodel = ResNet(Bottleneck, [2, 2, 2, 2],  chan_1=40, chan_2=60, chan_3=90, chan_4=135)
		print("model", bmodel)
	elif model_type == 'ResNet3':
		bmodel = ResNet(Bottleneck, [2, 2, 2, 2], chan_1=512)
		print("model", bmodel)
	elif model_type == 'Transformer':
		bmodel = Transformer(input_shape=(1, sig_len), nb_classes=2, mlp_units=[256], out_channels=24)
		print("model", bmodel)
	elif model_type == 'ResNet4Sequence':
		bmodel = ResNet4Sequence(Bottleneck4Sequence, [2, 2, 2, 2], num_letter = num_letter)
	else:
		raise ValueError('Invalid model option, choose either "resnet" or "transformer"')
	
	criterion = nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(bmodel.parameters(), lr=learningrate)

	# multiple GPU (optional)
	# if torch.cuda.device_count() > 1:
	# 	bmodel = nn.DataParallel(bmodel)
	bmodel = bmodel.to(device)

	if interm is not None:
		bmodel.load_state_dict(torch.load(interm))

	iteration = 0
  
	loss_epoch_train_plot = []
	loss_epoch_val_plot = []

	sensitivity_val_ep = []
	specificity_val_ep = []
	accuracy_val_ep = []
	precision_val_ep = []
	f1_val_ep = []
	
	learning_rates = []

	### Training
	for epoch in range(epoch):
		all_valy = []
		all_outputs_val = []

		total_loss_train = 0
		total_samples_train = 0
		total_loss_val = 0
		total_samples_val = 0

		for spx, spy in training_generator:
			if model_type == "ResNet4Sequence":
				spx = spx.to(torch.float)
			iteration += 1
			bmodel.train()
			spx, spy = spx.to(device), spy.to(torch.long).to(device)
			print(f"Signal tensor shape: {spx.shape}, Label tensor shape {spy.shape}")
			print(spx.dtype, spy.dtype)
			print(spx.device, spy.device)
			print(torch.unique(spy))
			
			outputs = bmodel(spx)
			# print("Output shape: ", outputs.shape)

			# average of the loss
			loss_training_ave = criterion(outputs, spy)
			# print("Training Loss step:", loss_training_ave.item())
			total_loss_train += loss_training_ave.item()
			total_samples_train += 1
   
			optimizer.zero_grad()
			loss_training_ave.backward()
			optimizer.step()

		# Calculate the average loss (for loss sum)
		average_loss_per_sample_train = total_loss_train / total_samples_train
		print(f"Average Loss per Sample (Train): {average_loss_per_sample_train}")
		# loss_epoch_train_plot.append(average_loss_per_sample_train)

		bmodel.eval()
		correct_preds_val, num_samples_val = 0, 0
		with torch.set_grad_enabled(False):
			for valx, valy in validation_generator:
				print(valx.shape, valy.shape)
				if model_type == "ResNet4Sequence":
					valx = valx.to(torch.float)
				valx, valy = valx.to(device), valy.to(torch.long).to(device)
				outputs_val = bmodel(valx)

				#average of the loss
				loss_val_ave = criterion(outputs_val, valy)
				# print("Validation Loss step:", loss_val_ave.item())
				total_loss_val += loss_val_ave.item()
				total_samples_val += 1

				correct_preds_val += (valy == outputs_val.max(dim=1).indices).sum().item()
				num_samples_val += valy.size(0)
				all_valy.extend(valy.cpu().numpy())
				all_outputs_val.extend(outputs_val.max(dim=1).indices.cpu().numpy())

			# Calculate the average loss
			average_loss_per_sample_val = total_loss_val / total_samples_val
			print(f"Average Loss per Sample (Val): {average_loss_per_sample_val}")

			tn, fp, fn, tp = confusion_matrix(all_valy, all_outputs_val).ravel()
			print(f"tp_val:{tp}, fn_val:{fn}, fp_val: {fp},tn_val:{tn}")

			acc_vt = correct_preds_val / num_samples_val
			sensitivity_val = tp / (tp + fn)
			specificity_val = tn / (tn + fp)
			precision_val = tp / (tp + fp)
			f1_val = f1_score(all_valy, all_outputs_val)

			accuracy_val_ep.append(acc_vt)
			sensitivity_val_ep.append(sensitivity_val)
			specificity_val_ep.append(specificity_val)
			precision_val_ep.append(precision_val)
			f1_val_ep.append(f1_val)

			print(f"F1 Score val: {f1_val}")
			print(f"Sensitivity val: {sensitivity_val}")
			print(f"Specificity val: {specificity_val}")
			print(f"Precision val: {precision_val}")
			print(f"Accuracy val: {acc_vt}")

			print("epoch: " + str(epoch+1) + ", iteration: " + str(iteration) + ", Curr. acc. validation: " + str(acc_vt))
		
		current_model_path = os.path.join(outpath, f"model_epoch_{epoch}.ckpt")
		torch.save(bmodel.state_dict(), current_model_path)

if __name__ == '__main__':
	main()
