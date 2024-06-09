import torch
import glob
import os
import argparse

def concat_tensor(pos_files, neg_files):
    pos_data = []
    for file in pos_files:
        pos_data.append(torch.load(file))
    pos_data_tensor = torch.cat(pos_data, dim=0)
    print(pos_data_tensor.shape)
    
    neg_data = []
    for file in neg_files:
        neg_data.append(torch.load(file))
    neg_data_tensor = torch.cat(neg_data, dim=0)
    print(neg_data_tensor.shape)
    
    return pos_data_tensor, neg_data_tensor

def save_tensor(pos_data_tensor, save_path_train_pos, neg_data_tensor, save_path_train_neg): 
    torch.save(pos_data_tensor, save_path_train_pos)
    torch.save(neg_data_tensor, save_path_train_neg)
    
def main(args):
    preprocessed_pos_files = glob.glob(os.path.join(args.preprocessed_pos_folder, '*.pt'))
    preprocessed_neg_files = glob.glob(os.path.join(args.preprocessed_neg_folder, '*.pt'))

    pos_tensor, neg_tensor = concat_tensor(preprocessed_pos_files, preprocessed_neg_files)
    save_tensor(pos_tensor, args.save_pos, neg_tensor, args.save_neg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate and save tensors.')
    
    parser.add_argument('--preprocessed_pos_folder', type=str, required=True, help='Path to positive preprocessed folder')
    parser.add_argument('--preprocessed_neg_folder', type=str, required=True, help='Path to negative preprocessed folder')
    
    parser.add_argument('--save_pos', type=str, required=True, help='Save path for concatenated training positive tensor')
    parser.add_argument('--save_neg', type=str, required=True, help='Save path for concatenated training negative tensor')
    
    args = parser.parse_args()
    
    main(args)
