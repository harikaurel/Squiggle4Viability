import torch
import torch as th
import os
import argparse

class OneHotEmbedding(th.nn.Module):
    """Embed inputs using one-hot encoding."""

    def __init__(self, num_embeddings: int):
        super().__init__()
        self.num_embeddings = num_embeddings

    def forward(self, x: torch.Tensor):
        """Return a one-hot encoded version of the given tensor.

        Shape:
            - x: `(N,)` for batched input, `()` for unbatched input.
            - output: `(N, E)` for batched input, `(E,)` for unbatched
              input.

            where N is the batch size, E is the number of
            embeddings/classes.
        """
        return th.nn.functional.one_hot(x, self.num_embeddings)

class BasesToTensor(torch.nn.Module):
    def __init__(self, bases_dict):
        super().__init__()
        self.bases_dict = bases_dict

    def forward(self, bases):
        return torch.tensor([self.bases_dict[base] for base in bases])

def read_sequences_from_txt(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            _, sequence = line.strip().split('\t')
            sequences.append(sequence)
    return sequences

def main(args):
    file_path = args.infile_path
    output_file = args.output_file
    bases = args.bases

    sequences = read_sequences_from_txt(file_path)

    bases_dict = {base: i for (i, base) in enumerate(bases)}
    num_embeddings = len(bases_dict)

    bases_to_tensor = BasesToTensor(bases_dict)
    one_hot_layer = OneHotEmbedding(num_embeddings=num_embeddings)

    one_hot_tensors = []
    for seq in sequences:
        sequence_tensor = bases_to_tensor(seq)
        one_hot_sequence = one_hot_layer(sequence_tensor)
        one_hot_tensors.append(one_hot_sequence)
    tensor_list_expanded = [t.unsqueeze(0) for t in one_hot_tensors]
    concat_tensor = torch.cat(tensor_list_expanded, dim=0)
    print(f"{output_file}_{concat_tensor.shape}")
    torch.save(concat_tensor, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a sequence file and save one-hot encoded tensor.')
    
    parser.add_argument('--infile_path', type=str, required=True, help='Path to the input .tsv file (first column: read_id, second column: sequence)')
    parser.add_argument('--output_file', type=str, required=True, help='File to save concatenated one-hot encoded tensor')
    parser.add_argument('--bases', type=str, required=True, help='String of bases for one-hot encoding (e.g., "ACGTM")')
    
    args = parser.parse_args()
    
    main(args)
