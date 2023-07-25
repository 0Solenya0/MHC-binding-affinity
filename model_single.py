import math
from torch import nn
import torch


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-', '*']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
START_TOKEN = AA_TO_IDX['*']
PEPTIDE_LEN = 15
EMB_SIZE = 32


def encode_peptide(peptide_str):
    return [START_TOKEN] + [AA_TO_IDX[peptide_str[i]] if i < len(peptide_str) else AA_TO_IDX['-'] for i in range(PEPTIDE_LEN - 1)]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=14):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.squeeze(1))

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, *args, **kwargs):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)

        self.kw, self.qw, self.vw = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(
            embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.ln_2 = nn.LayerNorm(embed_dim)

        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(self.qw(x), self.kw(x), self.vw(x))[0]
        x = self.ln_2(x)
        x = x + self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
        return x


class SingleModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SingleModel, self).__init__()
        self.pre = nn.Sequential(
            nn.Embedding(len(AMINO_ACIDS), EMB_SIZE),  # .from_pretrained(bl_embedding),
            PositionalEncoding(EMB_SIZE, max_len=PEPTIDE_LEN),
            # nn.Linear(24, 32),
            nn.Tanh(),
            *[Block(embed_dim=EMB_SIZE) for i in range(4)]
        )
        self.head_1 = nn.Sequential(
            nn.Linear(EMB_SIZE, 8),
            # nn.Linear(PEPTIDE_LEN * 8, 64),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 32),
            nn.ReLU()
        )
        self.head_2 = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pre(x)
        return self.sig(
            self.head_2(
                x[:, 0, :].squeeze(1) + self.head_1(x[:, 0, :].squeeze(1))
            )
        )
