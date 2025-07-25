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
from collections import defaultdict
from math import ceil



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
        outputs_test = bmodel(testx)

        # Determine the output file
        file_index = 0
        current_file = os.path.join(outpath, f'batch_{file_index}.txt')

        while count_lines(current_file) >= 4000:  # Check if current file exceeds the limit
            file_index += 1
            current_file = os.path.join(outpath, f'batch_{file_index}.txt')

        with open(current_file, 'a') as f:  # Append to the current file
            for nm, val, logits in zip(data_name, outputs_test.max(dim=1).indices.int().data.cpu().numpy(), outputs_test.data.cpu().numpy()):
                arr = [logits[0], logits[1]]
                tensor = torch.tensor(arr)
                pos_pred = (m(tensor).numpy())[1]  # Prints probability of being dead
                f.write(str(nm) + '\t' + str(val) + '\t' + str(pos_pred) + '\n')

        print(f"$$$$$$$$$$ Done processing with batch {batchi} to file {current_file}")
        del outputs_test


########################
#### Load the data #####
########################
def get_raw_data(inpath, fileNM, data_test, data_name):
	pod5_filepath = os.path.join(inpath, fileNM)
	with p5.Reader(pod5_filepath) as reader:
		for read in reader.reads():
			raw_data = read.signal
			if len(raw_data) >= 1500:
				data_test.append(raw_data[1500:])
				data_name.append(read.read_id)
	return data_test, data_name

# def get_raw_data(inpath, fileNM, data_test, data_name, read_ids_to_process):
# 	pod5_filepath = os.path.join(inpath, fileNM)
# 	with p5.Reader(pod5_filepath) as reader:
# 		for read in reader.reads():
# 			if read.read_id in read_ids_to_process:
# 				raw_data = read.signal
# 				data_test.append(raw_data)
# 				data_name.append(read.read_id)
# 	return data_test, data_name


def count_lines(filepath):
    """Count the number of lines in a file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return sum(1 for _ in f)
    return 0


@click.command()
@click.option('--model', '-m', help='The pretrained model path and name', type=click.Path(exists=True))
@click.option('--inpath', '-i', help='The input fast5 folder path', type=click.Path(exists=True))
@click.option('--outpath', '-o', help='The output result folder path', type=click.Path())
@click.option('--model_type', '-mt', default="ResNet1", help='Model to use for training, options are "ResNet1", "ResNet2", "ResNet3", "Transformer" or "ResNet4Sequence"')
# @click.option('--sig_len', '-sl', default=10000, help='chunk size of the signal')

def main(model, inpath, outpath, model_type):

	click.echo(f"""
    model: {model}
    inpath: {inpath}
    outpath: {outpath}
    model_type: {model_type}
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
		bmodel = Transformer(input_shape=(1, 10000), nb_classes=2, mlp_units=[256], out_channels=24)
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

	signal_groups = defaultdict(list)
	
	for fileNM in glob.glob(inpath + '/*.pod5', recursive=True):
		data_test = []
		data_name = []
		data_test, data_name = get_raw_data(os.path.dirname(fileNM), os.path.basename(fileNM), data_test, data_name)
		print(f"Processing file {fileNM} with {len(data_test)} reads")

		for raw_data, name in zip(data_test, data_name):
			signal_groups[len(raw_data)].append((raw_data, name))

	batchi = 0

	MAX_BATCH_SIZE = 1000

	for length, signals in signal_groups.items():
		print(f"Processing signals of length {length} with {len(signals)} entries")
		batch_data_test = [signal[0] for signal in signals]
		batch_data_name = [signal[1] for signal in signals]

		if length > 10000:
			# Process each signal individually
			for raw_data, name in zip(batch_data_test, batch_data_name):
				print(f"Processing individual signal with length {length} (batch {batchi})")
				signal = np.array(raw_data).reshape(1, -1)
				signal = normalization(signal, batchi, length)
				process(signal, [name], batchi, bmodel, outpath, device)
				batchi += 1
		else:
			# Process smaller signals in batches
			num_batches = ceil(len(batch_data_test) / MAX_BATCH_SIZE)
			for batch_index in range(num_batches):
				start_idx = batch_index * MAX_BATCH_SIZE
				end_idx = min(start_idx + MAX_BATCH_SIZE, len(batch_data_test))

				batch_chunk_test = batch_data_test[start_idx:end_idx]
				batch_chunk_name = batch_data_name[start_idx:end_idx]

				if len(batch_chunk_test) > 1:
					print(f"Processing batch {batchi} with length {length} ({start_idx} to {end_idx})")
					batch_chunk_test = normalization(np.array(batch_chunk_test), batchi, length)
					process(batch_chunk_test, batch_chunk_name, batchi, bmodel, outpath, device)
				else:
					# Process single signal individually
					print(f"Processing batch {batchi} with length {length} ({start_idx} to {end_idx})")
					signal = np.array(batch_chunk_test).reshape(1, -1)
					signal = normalization(signal, batchi, length)
					process(signal, batch_chunk_name, batchi, bmodel, outpath, device)

				print(f"$$$$$$$$$$ Done with batch {batchi} for signals of length {length}")
				batchi += 1




if __name__ == '__main__':
	main()
