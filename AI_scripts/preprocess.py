import os
import glob
import click
import torch
import numpy as np
from scipy import stats
import pod5 as p5
import torch

def normalization(data_test, xi, outpath, sig_len, pos = True):
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
	if pos is True:
		torch.save(torch.tensor(data_test).float(), outpath + '/pos_' + str(xi) + '.pt')
	else:
		torch.save(torch.tensor(data_test).float(), outpath + '/neg_' + str(xi) + '.pt')

@click.command() 
@click.option('--inpath', '-i', required=True, help='The input pod5 directory path')
@click.option('--outpath', '-o', required=True, help='The output tensor directory path')
@click.option('--datatype', '-dt', required=True, help='pos (target) or neg (non-target)')
@click.option('--sig_len', '-sl', default=10000, help='signal length, default 1000')

def main(inpath, outpath, datatype, sig_len):

	click.echo(f"""
    inpath: {inpath}
    outpath: {outpath}
    datatype: {datatype}
    sig_len: {sig_len}
    """)

	if datatype == "pos":
		pos = True
	else:
		pos = False
		
	os.makedirs(outpath, exist_ok=True)

	bi = 0

	batch_size = 1000  # Define the batch size

	for fileNM in glob.glob(inpath + '/*.pod5'):
		print(f"##### Processing file: {fileNM}")
		
		arr = []  # Reset the signal array for each file

		with p5.Reader(fileNM) as reader:
			for read in reader.reads():
				raw_data = read.signal
				arr.append(raw_data)
				bi += 1
				
				# Process the batch when the batch size is reached
				if len(arr) == batch_size:
					print(f"Processing batch of size {len(arr)} for file {fileNM} (signals processed until now: {bi})")
					normalization(arr, bi, outpath, sig_len, pos)
					arr = []  # Reset the array for the next batch

		# Process the remaining signals (if any) after finishing the current file
		if arr:
			print(f"Processing final batch of size {len(arr)} for file {fileNM} (signals processed until now: {bi})")
			normalization(arr, bi, outpath, sig_len, pos)

if __name__ == '__main__':
	main()
