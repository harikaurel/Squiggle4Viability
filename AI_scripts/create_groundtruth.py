import os
import glob
import pod5 as p5
import pandas as pd
import numpy as np
import argparse

def get_read_label(directory, label="alive"):
    read_label = []
    for input_file in glob.glob(os.path.join(directory, '**', '*.pod5'), recursive=True):
        with p5.Reader(input_file) as reader:
            read_label.extend([[read.read_id, label] for read in reader.reads()])
    return read_label

def groundtruth_df(directory, label="alive"):
    read_label = get_read_label(directory, label)
    return pd.DataFrame(read_label, columns=['read_id', 'label'])

def prepare_df(df, num_train, num_val, num_test):
    df['read_id'] = df['read_id'].astype(str)
    df['dataset_type'] = np.nan
    df.iloc[:num_train, df.columns.get_loc('dataset_type')] = 'training'
    df.iloc[num_train:num_train+num_val, df.columns.get_loc('dataset_type')] = 'validation'
    df.iloc[num_train+num_val:num_train+num_val+num_test, df.columns.get_loc('dataset_type')] = 'testing'
    df.iloc[num_train+num_val+num_test:, df.columns.get_loc('dataset_type')] = 'remaining'
    
    # Check for duplicated read_ids and remove them
    df['read_id_first_two_parts'] = df['read_id'].apply(lambda x: '-'.join(x.split('-')[:2]))
    grouped = df.groupby('read_id_first_two_parts')['dataset_type'].nunique()
    overlapping_read_ids = grouped[grouped > 1].index.tolist()
    df = df[~df['read_id'].str.startswith(tuple(overlapping_read_ids))]
    
    return df

def get_signals(df, dataset_type):
    return df[df['dataset_type'] == dataset_type]['read_id']

def balance_and_limit_signals(train_alive, val_alive, test_alive, train_dead, val_dead, test_dead):
    min_train_length = min(len(train_alive), len(train_dead))
    min_train_length = round(min_train_length, -3)
    
    min_val_test_length = min(len(val_alive), len(test_alive), len(val_dead), len(test_dead))
    min_val_test_length = round(min_val_test_length, -3)
    
    train_alive = train_alive[:min_train_length]
    train_dead = train_dead[:min_train_length]
    val_alive = val_alive[:min_val_test_length]
    val_dead = val_dead[:min_val_test_length]
    test_alive = test_alive[:min_val_test_length]
    test_dead = test_dead[:min_val_test_length]
    
    return train_alive, val_alive, test_alive, train_dead, val_dead, test_dead

def main(alive_dir, dead_dir, output_dir):
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Create groundtruth dataframes
    df_read_ctrl_alive = groundtruth_df(alive_dir, label="alive")
    df_read_uv_dead = groundtruth_df(dead_dir, label="dead")

    # Calculate the number of samples
    num_viable = len(df_read_ctrl_alive)
    num_dead = len(df_read_uv_dead)
    ratio = 1
    num_samples_viable = min(num_viable, num_dead * ratio)
    num_samples_dead = min(num_dead, num_viable // ratio)

    # Determine dataset splits
    num_signals = num_samples_viable
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2

    num_train = int(num_signals * train_ratio)
    num_val = int(num_signals * val_ratio)
    num_test = int(num_signals * test_ratio)

    # Prepare dataframes
    df_read_ctrl_alive = prepare_df(df_read_ctrl_alive, num_train, num_val, num_test)
    df_read_uv_dead = prepare_df(df_read_uv_dead, num_train, num_val, num_test)

    # Get signals for each dataset type
    train_signals_viable = get_signals(df_read_ctrl_alive, 'training')
    val_signals_viable = get_signals(df_read_ctrl_alive, 'validation')
    test_signals_viable = get_signals(df_read_ctrl_alive, 'testing')

    train_signals_dead = get_signals(df_read_uv_dead, 'training')
    val_signals_dead = get_signals(df_read_uv_dead, 'validation')
    test_signals_dead = get_signals(df_read_uv_dead, 'testing')

    # Balance and limit signal counts
    train_signals_viable, val_signals_viable, test_signals_viable, train_signals_dead, val_signals_dead, test_signals_dead = balance_and_limit_signals(
        train_signals_viable, val_signals_viable, test_signals_viable,
        train_signals_dead, val_signals_dead, test_signals_dead
    )

    # Output signal counts
    print(len(train_signals_viable), len(val_signals_viable), len(test_signals_viable))
    print(len(train_signals_dead), len(val_signals_dead), len(test_signals_dead))

    # Write ground truth files
    dead_list = [train_signals_dead, val_signals_dead, test_signals_dead]
    dead_gt_filenames = ["train_dead.txt", "val_dead.txt", "test_dead.txt"]
    viable_list = [train_signals_viable, val_signals_viable, test_signals_viable]
    viable_gt_filenames = ["train_alive.txt", "val_alive.txt", "test_alive.txt"]

    for dataset, gt_filename in zip(dead_list, dead_gt_filenames):
        print(len(dataset))
        dataset.to_csv(os.path.join(output_dir, gt_filename), sep='\t', index=False)

    for dataset, gt_filename in zip(viable_list, viable_gt_filenames):
        print(len(dataset))
        dataset.to_csv(os.path.join(output_dir, gt_filename), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pod5 files and create datasets.')
    parser.add_argument('--alive_dir', type=str, required=True, help='Directory for alive pod5 files')
    parser.add_argument('--dead_dir', type=str, required=True, help='Directory for dead pod5 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for ground truth files')

    args = parser.parse_args()
    main(args.alive_dir, args.dead_dir, args.output_dir)
