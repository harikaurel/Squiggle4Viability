import math
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
import pod5
from ResNet import ResNet, Bottleneck
from inference import normalization
import evaluation_functions

# Instantiate the model and load its weights
def return_model(model_weights=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet(Bottleneck, [2, 2, 2, 2])
    model.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))
    model = model.to(device).eval()
    return model

# Get raw data from pod5 files
def get_raw_data(pod5_file, data_test, data_name):
    with pod5.Reader(pod5_file) as reader:
        for read in reader.reads():
            raw_data = read.signal
            data_test.append(raw_data)
            data_name.append(read.read_id)
    return data_test, data_name

def get_chunks(pod5_file):
    data_test = []
    data_name = []
    data_test, data_name = get_raw_data(pod5_file, data_test, data_name)
    data_test = normalization(data_test, 0, 10000)
    yield data_name, data_test, pod5_file

# Normalize the CAM between 0 and 1
def normalize_scoremap(cam):
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

# Get the numpy array of a tensor
def t2n(t):
    return t.detach().cpu().numpy().astype(float)

# Generate the CAM for a specific sample
def compute_cams(model, X_test, y_test, height=1, width=10000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, cams = model(torch.unsqueeze(X_test, 0).to(device), y_test, True)
    x = nn.Softmax(dim=-1)(x)
    cams = t2n(cams)
    cam_resized = cv2.resize(cams, (width, height), interpolation=cv2.INTER_CUBIC)
    cam_normalized = normalize_scoremap(cam_resized)
    corr_pred = torch.argmax(x) == y_test
    prob = 0.5
    if corr_pred and torch.max(x) < prob:
        corr_pred = False
    return cam_normalized, corr_pred

# Plot and save CAM maps
def plot_maps(key, value, cam_folder, pod5_file):
    pod_file = os.path.basename(pod5_file).split(".")[0]
    cam_folder = os.path.join(cam_folder, pod_file)
    os.makedirs(cam_folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 5))
    cam = value[0]
    pcm = ax.imshow(cam, cmap="Reds", aspect="auto", extent=[0, 10000, math.floor(value[-1].min()), math.ceil(value[-1].max())], alpha=1.0)
    ax.set_title(key)
    ax.plot(value[-1], color="black")
    fig.colorbar(pcm, ax=ax, shrink=0.6)
    plt.savefig(os.path.join(cam_folder, f"{key}.pdf"))
    plt.close(fig)

def generate_CAM_4_pod5(tn_buf, pod5_file, cam_folder="death_cams"):
    for key, value in tn_buf.items():
        plot_maps(key, value, cam_folder, pod5_file)

def main(args):
    groundtruth_dict = evaluation_functions.get_groundtruthdict(args.ground_truth_file, label_type="pos")
    model = return_model(args.model_weights)
    for chunk_names, chunk_tensors, pod5_file in get_chunks(args.pod5_path):
        t_class = {}
        for idx in tqdm(range(len(chunk_names))):
            chunk_name = str(chunk_names[idx])
            chunk_tensor = chunk_tensors[idx]
            y = groundtruth_dict.get(chunk_name, None)
            if y is not None:
                mapp, tpred = compute_cams(model, chunk_tensor, y)
                if tpred:
                    t_class[chunk_name] = [mapp, chunk_tensor]
        generate_CAM_4_pod5(t_class, pod5_file, args.cam_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, required=True)
    parser.add_argument("--model_weights", type=str, default="ResNet_677ep.ckpt")
    parser.add_argument("--cam_folder", type=str, required=True, help="output folder to save the cams")
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--pod5_path", type=str, required=True, help="path to the specific pod5 file")
    args = parser.parse_args()
    main(args)
