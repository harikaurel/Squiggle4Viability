import math
from pathlib import Path
import os
import sys
import pickle
import cv2
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import argparse
from ResNet import ResNet, Bottleneck
from inference import get_raw_data, normalization

from torch.utils.data import Dataset, DataLoader

class SquiggleData(Dataset):
    def __init__(self, chunk_names, data, labels):
        self.chunk_names = chunk_names
        self.data = data
        self.labels = labels

    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx):
        return str(self.chunk_names[idx]), self.data[idx], torch.tensor(self.labels[idx])


# Instanciante the model and load its weights
def get_model(model_weights=""): 
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = ResNet(Bottleneck, [2, 2, 2, 2])
    model.load_state_dict(torch.load(model_weights, map_location=torch.device(device)))
    model = model.to(device).eval()
    return model


# Return the read_ids and their corresponding tensors for each pod5 file
def get_chunks(pod_folder):
    it = 0
    batchi = 0
    for ipod in tqdm(glob.glob(pod_folder + "/*.pod5")):
        data_test = []
        data_name = []

        data_test, data_name = get_raw_data(pod_folder, ipod, data_test, data_name)
        it += 1
        data_test = normalization(data_test, batchi, 10000)

        yield data_name, data_test, ipod

# Normalize the CAM between 0 and 1
def normalize_scoremap(cam):
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)

    cam -= cam.min()
    cam /= cam.max()

    return cam

# from numpy array to tensor
def t2n(t):
    return t.detach().cpu().numpy().astype(float)

def remove_max_region(cam, signal, mask_values=100):
    # Get the maximum value of the CAM of each sample
    indices = cam.argmax(axis=1)

    mask_values = mask_values // 2
    
    arg_max = indices + mask_values
    arg_max[arg_max>=cam.shape[1]] = cam.shape[1] - 1
    arg_max = torch.from_numpy(arg_max).unsqueeze(1)

    arg_min = indices - mask_values
    arg_min[arg_min<0] = 0
    arg_min = torch.from_numpy(arg_min).unsqueeze(1)

	# Create a range cam that matches the indices
    range_cam = torch.arange(cam.shape[1]).expand_as(torch.tensor(cam))

	# Create masks by comparing range_cam with arg_min and arg_max
    mask = (range_cam >= arg_min) & (range_cam <= arg_max)
    signal[mask] = 0

    return signal

# %% # Generate the CAM for each sample
def compute_cams(model, X_test, y_test, width=10000, minimum_prob=0.5):
    
    # Get the logits as well as the CAM for each sample
    x, cams = model(X_test, y_test, True)
    x = nn.Softmax(dim=-1)(x)
    cams = t2n(cams)

    # Resize the CAM to the same size of the original signal and normalize the values
    cam_resized = cv2.resize(cams, (width, X_test.size()[0]), interpolation=cv2.INTER_CUBIC)
    cam_normalized = normalize_scoremap(cam_resized)

    x_argmax = torch.argmax(x, dim=1)
    
    corr_pred = (torch.max(x, dim=1).values > minimum_prob) & (x_argmax == y_test)
    corr_pred = corr_pred.cpu().numpy()
    
    cam_normalized = cam_normalized

    # Return the normalized CAM and a bool if the predicted label is correct or not
    return cam_normalized, corr_pred, X_test, x, x_argmax


# %%
def plot_maps(cam, signal, chunk_name, folder, pod_file):
    # Simplify directory creation
    pod_folder = os.path.join(folder, pod_file)
    os.makedirs(pod_folder, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Plotting the CAM
    pcm = ax.imshow(
        np.expand_dims(cam, axis=0),
        cmap="Reds",
        aspect="auto",
        extent=[0, 10000, math.floor(signal.min()), math.ceil(signal.max())],
        alpha=1.0,
    )
    ax.set_title(chunk_name)

    # Plotting the corresponding signal
    ax.plot(signal.detach().cpu().numpy().astype(float), color="black")

    # Adding colorbar and saving the plot
    fig.colorbar(pcm, ax=ax, shrink=0.6)
    
    plt.savefig(os.path.join(pod_folder, f"{chunk_name}.png"))
    plt.close(fig)

def plot_sequential_masks(chunk_name, cam, signal, prob, pod_file, folder="CAMs_with_mask", masked_value=100):

    fig, ax = plt.subplots(6, 1, figsize=(15, 30))

    for i in range(6):

        pcm = ax[i].imshow(
            np.expand_dims(cam[i], axis=0),
            cmap="Reds",
            aspect="auto",
            extent=[0, 10000, math.floor(min(signal[i])), math.ceil(max(signal[i]))],
            alpha=1.0,
        )

        ax[i].set_title(f"{chunk_name} Masked {masked_value} values / prob {prob[i]:.2f}, Label Death")

        # Plotting the corresponding signal
        ax[i].plot(signal[i], color="black")
        fig.colorbar(pcm, ax=ax[i], shrink=0.6)
    
    folder_name = os.path.join(folder, pod_file)
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, f"{chunk_name}.png"))
    plt.close(fig)


def get_nbr_dark_region(cams, thr="0.8", nbr_dark_regions=2):

    valid_positions = {"maxi", "mini"}
    if thr not in valid_positions and not (0.0 <= float(thr) <= 1.0):
        print("Error: position must be 'maxi', 'mini', or a number between 0.0 and 1.0")
        return

    arg = float(thr) if thr not in valid_positions else None

    if thr == "maxi":
        arg = np.argmax(cams, axis=1) 
    elif thr == "mini":
        arg = np.argmin(cams, axis=1) 
    else:
        arg = float(thr)

    # Get the dark regions on CAMs where 1 is a dark region because the value is greater than 
    # the threshold and 0 is a light region
    cams_dark_regions = (cams >= arg).astype(int)

    # Add a padding of 0 at the end of each CAM to be able to spot the dark region at the end of the CAM
    cams_dark_regions_pad = np.pad(cams_dark_regions, ((0, 0), (0, 1)), mode='constant', constant_values=False)
   
    # Get the transition between the dark and light regions to be able 
    # to calculate the number of dark regions in the CAM
    # -1 is a transition from dark to light and 1 is a transition from light to dark  
    transition = np.diff(cams_dark_regions_pad, axis=1)
    
    # Get the sum of dark regions in the CAM
    sum_of_dark_regions = np.sum(transition==-1, axis=1)
    
    # Get the CAMs where there is only one dark region
    sum_of_dark_regions = sum_of_dark_regions == nbr_dark_regions

    return sum_of_dark_regions


def main(args):

    model = get_model(args.model_weights)

    # We store the probabilities of the samples without masking and after each 5 masking step in a list
    all_probs = [[], [], [], [], [], []]
    
    for chunk_names, chunk_tensors, ipod_file in tqdm(get_chunks(args.data_folder)):
        true_pos = dict()
        pod_file = os.path.basename(ipod_file).split(".")[0]
    
        print(f"Processing pod file: {pod_file}")

        test_data = SquiggleData(chunk_names, chunk_tensors, [1]*len(chunk_tensors))
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        for chunk_names, data, labels in tqdm(test_loader):

            data = data.to("cuda")
            labels = labels.long().to("cuda")

            cam, tpred, data, prob, labels = compute_cams(model, data, labels, minimum_prob=float(args.prob_thr))
            
            if tpred.sum() == 0:
                continue

            dark_region = np.ones_like(tpred[tpred], dtype=bool)

            if args.regions:
                dark_region = get_nbr_dark_region(cam[tpred], args.thr, args.nbr_dark_region)

                if dark_region.sum() == 0:
                    continue
     
            cam = cam[tpred][dark_region]
            data = data[tpred][dark_region]
            labels = labels[tpred][dark_region]
            prob = prob[tpred][:,1][dark_region]
            chunk_names = np.array(chunk_names)[tpred][dark_region]

            all_probs[0].extend(prob.tolist())

            cams_steps = [cam]
            data_steps = [data.detach().cpu().numpy().astype(float)]
            prob_steps = [prob.detach().cpu().numpy().astype(float)]
            
            for mask_idx in range(1, 6):
                data = remove_max_region(cam, data, args.mask)
                cam, tpred, data, prob, _ = compute_cams(model, data, labels)
                
                all_probs[mask_idx].extend(prob[:,1].tolist())

                cams_steps.append(cam.copy())
                data_steps.append(data.detach().cpu().numpy().astype(float).copy()) 
                prob_steps.append(prob[:,1].detach().cpu().numpy().astype(float))

            cams_steps = np.stack(cams_steps, axis=1).tolist()
            data_steps = np.stack(data_steps, axis=1).tolist()
            prob_steps = np.stack(prob_steps, axis=1).tolist()
            
            chunk_names = chunk_names.tolist()

            list(map(lambda kv: plot_sequential_masks(kv[0], kv[1], kv[2], kv[3], pod_file, args.cams_folder_name, str(args.mask)), zip(chunk_names, cams_steps, data_steps, prob_steps)))


    print("Saving the probabilities")
    print(len(all_probs[0]))

    # Create the folder to save the CAMs
    folder_name = "masked"+"_"+str(args.mask)+"_values"
    os.makedirs(folder_name, exist_ok=True)
    file_name = os.path.join(folder_name, "probs.pkl")
    
    with open(file_name, "wb") as f:
        pickle.dump(all_probs, f)
    


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, default="path/to/ground_truth_file")
    parser.add_argument("--data_folder", type=str, default="path/to/pod5_folder")
    parser.add_argument("--model_weights", type=str, default="ResNet_677ep.ckpt")
    parser.add_argument(
        "--cams_folder_name",
        type=str,
        default="CAMs",
        help="folder to save the cams",
    )
    parser.add_argument( "--mask", type=int, default=200, help="length of the sequence to be masked")
    parser.add_argument(
        "--thr",
        type=str,
        default="0.8",
        help="CAM threshold, choose between maxi, mini or a number between 0.0 and 1.0",
    )
    parser.add_argument(
        "--prob_thr",
        type=str,
        default="0.99",
        help="return signal chunk with a probability greater than the threshold",
    )
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--nbr_dark_region", type=int, default=1)
    parser.add_argument("--regions", type=bool, default=True)
    args = parser.parse_args()

    print(args)
    main(args)
