import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt

def plot(stats, colors=['skyblue', 'salmon'], values='p50'):
    plt.subplots(figsize=(10, 5))
    
    for i, (mean_with_std, size) in enumerate(stats):
        # Unpacking mean, std_devs, size
        mean, std_devs = zip(*mean_with_std)

        # Recalculate SEMs and CIs inside the plotting function
        sems = [sd/size for sd in std_devs]

        ci_95 = 1.96 * np.asarray(sems)

        # Create the line plot with 95% CI shaded area
        plt.plot(range(len(mean)), mean, marker='o', label=f'{values[i]} signals', linewidth=0.5)
        plt.fill_between(range(len(mean)), np.array(mean) - ci_95, np.array(mean) + ci_95, color=colors[i], alpha=0.5, edgecolor='none')

    plt.xlabel('XAI Masking Steps')
    plt.ylabel('Prediction Probability')

    # Removing the top and right frame lines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), frameon=False)

    # Save the plot as a PNG file and a PDF file
    plt.savefig(f'distribution_plots_all_p50.png', dpi=300)
    plt.savefig(f'distribution_plots_all_p50.pdf', dpi=300)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=list, default=["masked_2000_values", "masked_1000_values", "masked_400_values", "masked_200_values", "masked_100_values"], help="Folder names where probabilities are stored")
    parser.add_argument("--values", type=list, default=['2000', '1000', '400', '200', "100"], help="Masked values that we use for label")
    parser.add_argument("--title", type=str, default='distribution_plots_all_p50.png', help="Title of the plot")
    
   
    args = parser.parse_args()
    print(args)
   
    if len(args.files) != len(args.values):
        raise ValueError("Number of files and values should be the same")

    colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    all_means_std = []  
    for file in args.files:
        print(f"Processing {file}")
        
        with open(f"{file}/probs.pkl", 'rb') as k:
            masked = pickle.load(k)

        mean_std = []
        for probs in masked:
            mean = np.mean(probs)
            std = np.std(probs)
            size = len(probs)
            mean_std.append((mean, std))
        all_means_std.append((mean_std, size))

    plot(all_means_std, colors, args.values)

