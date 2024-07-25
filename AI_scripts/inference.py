import time
import torch
import click
import os
import glob
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F
from scipy import stats
from ResNet import ResNet
from ResNet import Bottleneck
from ResNet4Sequence import ResNet4Sequence
from ResNet4Sequence import Bottleneck4Sequence
from Transformer import Transformer
import pod5 as p5

########################
##### Normalization ####
########################
def normalization(data_test, batchi, sig_len):
	mad = stats.median_abs_deviation(data_test, axis=1, scale='normal')
	m = np.median(data_test, axis=1)
	data_test = ((data_test - np.expand_dims(m,axis=1))*1.0) / (1.4826 * np.expand_dims(mad,axis=1))

	x = np.where(np.abs(data_test) > 3.5)

	for i in range(x[0].shape[0]):
		if x[1][i] == 0:
			data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]+1]
		elif x[1][i] == sig_len-1:
			data_test[x[0][i],x[1][i]] = data_test[x[0][i],x[1][i]-1]
		else:
			data_test[x[0][i],x[1][i]] = (data_test[x[0][i],x[1][i]-1] + data_test[x[0][i],x[1][i]+1])/2

	data_test = torch.tensor(data_test).float()

	print(f"$$$$$$$$$$ Done data normalization with batch {batchi}")
	return data_test


########################
####### Run Test #######
########################
def process(data_test, data_name, batchi, bmodel, outpath, device):
	m = nn.Softmax(dim=0)
	with torch.no_grad():
		testx = data_test.to(device)
		# print("testx", testx)
		outputs_test = bmodel(testx)
		# print("output", outputs_test)
		with open(outpath + '/batch_' + str(batchi) + '.txt', 'w') as f:
			for nm, val, logits in zip(data_name, outputs_test.max(dim=1).indices.int().data.cpu().numpy(), outputs_test.data.cpu().numpy()):
				arr = [logits[0], logits[1]]
				tensor = torch.tensor(arr)
				pos_pred = (m(tensor).numpy())[1] # prints probability of being dead
				f.write(str(nm) + '\t' + str(val) + '\t' + str(pos_pred) + '\n')
		print("$$$$$$$$$$ Done processing with batch " + str(batchi))
		del outputs_test

########################
#### Load the data #####
########################
def get_raw_data(inpath, fileNM, data_test, data_name):
	pod5_filepath = os.path.join(inpath, fileNM)
	with p5.Reader(pod5_filepath) as reader:
		for read in reader.reads():
			raw_data = read.signal
			data_test.append(raw_data)
			data_name.append(read.read_id)
	return data_test, data_name

@click.command()
@click.option('--model', '-m', help='The pretrained model path and name', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='The input fast5 folder path', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='The output result folder path', type=click.Path())
@click.option('--model_type', '-mt', default="ResNet1", help='Model to use for training, options are "ResNet1", "ResNet2", "ResNet3", "Transformer" or "ResNet4Sequence"')
@click.option('--sig_len', '-sl', default=10000, help='chunk size of the signal')

def main(model, inpath, outpath, model_type, sig_len):

	click.echo(f"""
    model: {model}
    inpath: {inpath}
    outpath: {outpath}
    model_type: {model_type}
    sig_len: {sig_len}
    """)
	
	start_time = time.time()
	if torch.cuda.is_available():
		device = torch.device('cuda')
		print(device)
		print(torch.cuda.device_count())
	else:
		device = torch.device('cpu')
	
	# make output folder
	if not os.path.exists(outpath):
		os.makedirs(outpath)

	# load model

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

	saved_state_dict = torch.load(model, map_location=device)

	has_module = any("module." in k for k in saved_state_dict.keys())

	if torch.cuda.device_count() > 1:
		if not has_module:
			saved_state_dict = {"module." + k: v for k, v in saved_state_dict.items()}
		bmodel = nn.DataParallel(bmodel)
		bmodel = bmodel.to(device).eval()
		bmodel.load_state_dict(saved_state_dict)
  
	else:
		if has_module:
			saved_state_dict = {k.replace("module.", ""): v for k, v in saved_state_dict.items()}

		bmodel = bmodel.to(device).eval()
		bmodel.load_state_dict(saved_state_dict)

	print("$$$$$$$$$$ Done loading model")

	batchi = 0

	for fileNM in glob.glob(inpath + '/**/*.pod5', recursive=True):
		data_test = []
		data_name = []
		file_directory = os.path.dirname(fileNM)
		data_test, data_name = get_raw_data(file_directory, os.path.basename(fileNM), data_test, data_name)
		data_test = normalization(data_test, batchi, sig_len)
		process(data_test, data_name, batchi, bmodel, outpath, device)
		print(f"$$$$$$$$$$ Done with batch {batchi}\n")
		batchi += 1

	print("FINAL--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
	main()
