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
@click.option('--batch', '-b', default=1000, help='Batch size, default 1000')
@click.option('--sig_len', '-sl', default=10000, help='signal length, default 1000')

def main(inpath, outpath, datatype, batch, sig_len):

	click.echo(f"""
    inpath: {inpath}
    outpath: {outpath}
    datatype: {datatype}
    batch: {batch}
    sig_len: {sig_len}
    """)

	if datatype == "pos":
		pos = True
	else:
		pos = False
		
	os.makedirs(outpath, exist_ok=True)

	arr = []
	bi = 0
	
	for fileNM in glob.glob(inpath + '/*.pod5'):
		print("##### file: " + fileNM)
		with p5.Reader(fileNM) as reader:
			for read in reader.reads():
				raw_data = read.signal
				bi += 1
				arr.append(raw_data)
				if (bi%batch == 0) and (bi != 0):
					normalization(arr, bi, outpath, sig_len, pos)
					del arr
					arr = []


if __name__ == '__main__':
	main()
