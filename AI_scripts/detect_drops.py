import argparse
import sys
import os
import numpy as np
import pod5
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats


def normalization(data_test):
    mad = stats.median_abs_deviation(data_test, scale='normal')
    m = np.median(data_test)
    normalized_data = ((data_test - m) * 1.0) / (1.4826 * mad)
    outliers = np.abs(normalized_data) > 3.5
    # Handle outliers by assigning the median of the nearby points
    normalized_data[outliers] = np.median(normalized_data[~outliers])
    return normalized_data


def detector(sig, std_scale=3):
    mean = np.mean(sig)
    stdev = np.std(sig)
    threshold = mean - (stdev * std_scale)
    return np.where(sig <= threshold)[0], mean, stdev, threshold  # Directly return indices where condition is met


def view_segs(cands, mean, bot, sig, readID, args):
    '''
    View the points in signal
    '''
    # candidate padding around detected position for nice visualisation
    pad = args.padding
    # set up plot
    fig, ax = plt.subplots()
    plt.title(readID)

    # plot the signal
    ax.plot(sig, color='k')

    # Show candidate regions using padding
    for c in cands:
        ax.axvspan(c - pad, c + pad, alpha=0.5, color='orange')

    # plot the mean and the bot threshold
    ax.axhline(y=mean, color='green')
    ax.axhline(y=bot, color='red')
    
    # Ensure figure path exists
    os.makedirs(args.figure_path, exist_ok=True)
    
    # Save the plot as a PDF file
    plt.savefig(os.path.join(args.figure_path, f"{readID}.pdf"))
    plt.close(fig)


def process_pod5_file(in_path, out_path, args):
    with open(out_path, 'a') as file:
        with pod5.Reader(in_path) as reader:
            for read in reader.reads():
                readID = read.read_id
                sig = read.signal
                normalized_sig = normalization(sig)
                cands, mean, stdev, threshold = detector(normalized_sig)
                if len(cands) > 0:
                    sudden_drop = "True"
                    view_segs(cands, mean, threshold, normalized_sig, readID, args)
                else:
                    sudden_drop = "False"

                file.write(f"{readID}\t{sudden_drop}\t{len(cands)}\t{len(normalized_sig)}\n")
                print(f"{readID}\t{sudden_drop}\t{len(cands)}\n")


def process_folder(in_path, out_path, args):
    for filename in os.listdir(in_path):
        if filename.endswith(".pod5"):
            file_path = os.path.join(in_path, filename)
            print(f"Processing {file_path}")
            process_pod5_file(file_path, out_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process POD5 files for nanopore signal analysis.")
    parser.add_argument("--in_path", type=str, required=True, help="The path to the folder containing .pod5 files to be analyzed")
    parser.add_argument("--out_path", type=str, required=True, help="The path for the output TSV file")
    parser.add_argument("--figure_path", type=str, required=True, help="Path to save figures")
    parser.add_argument("--padding", type=int, default=50, help="Padding around detected candidates for visualization")

    args = parser.parse_args()

    process_folder(args.in_path, args.out_path, args)
    print(f"args: {args}", file=sys.stderr)
