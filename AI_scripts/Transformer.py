import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_positional_encoding(seq_length, d_model):
    position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, ff_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, d_model, kernel_size=1),
        )

    def forward(self, x):
        res = x
        # print("x before norm1", x.shape)
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print("after norm1 x", x.shape)
        x = x.permute(0, 2, 1)
        # print("before attention x", x.shape)
        x, _ = self.attn(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x + res
        res = x
        # print("before norm2 x", x.shape)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print("after norm2 x", x.shape)
        x = self.ff(x)
        x = x + res
        return x

class Transformer(nn.Module):
    def __init__(self, input_shape, nb_classes, num_heads=1, ff_dim=256,
             num_transformer_blocks=1, mlp_units=None, dropout=0.1, mlp_dropout=0.25, in_channels=1, out_channels=1, kernel_size=30, d_model=1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        # self.conv_name = f"conv{out_channels}"
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=10, padding=0)
        self.positional_encoding = get_positional_encoding(input_shape[1], input_shape[0])
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=out_channels, nhead=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        self.mlp = nn.Sequential()
        for i, dim in enumerate(mlp_units):
            self.mlp.add_module('fc{}'.format(i), nn.Linear(out_channels, dim))
            self.mlp.add_module('relu{}'.format(i), nn.ReLU())
            self.mlp.add_module('dropout{}'.format(i), nn.Dropout(mlp_dropout))
        self.classifier = nn.Linear(mlp_units[-1], nb_classes)
    
    def forward(self, x, add_convolution=True):
        x = x.unsqueeze(1)
        positional_encoding = get_positional_encoding(x.size(2), self.d_model)
        positional_encoding = positional_encoding.transpose(1, 2).to(x.device)
        x = x + positional_encoding
        if add_convolution:
            x = self.conv(x)
            x = self.relu(x)
        else:
            print('no convolution')
        for block in self.blocks:
            x = block(x)
        x = F.avg_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        x = self.mlp(x)
        x = self.classifier(x)
        return x