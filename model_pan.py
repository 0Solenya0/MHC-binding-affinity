import math
from torch import nn
import torch

AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
               'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-', '*']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
MHC_POSITION_MASK_IND = [1, 3, 8, 9, 13, 16, 20, 32, 34, 35, 47, 66, 68,
                         85, 86, 88, 89, 90, 92, 93, 94, 97, 99, 100, 102, 103,
                         105, 106, 118, 120, 126, 128, 132, 136, 137, 139, 154, 161, 174,
                         175, 179, 186, 205, 207, 212, 216, 217, 230, 262, 276, 303, 305,
                         311, 315, 316, 317, 318, 319, 320, 321, 322, 328, 330, 331, 334,
                         337, 338, 344, 348, 357, 362, 363, 364]
MHC_TYPES = ['HLA-B*27:05', 'HLA-B*15:01', 'HLA-A*31:01', 'HLA-B*18:01', 'HLA-B*57:01', 'HLA-C*02:02', 'HLA-B*35:01', 'HLA-A*33:01', 'HLA-B*15:42', 'HLA-A*02:06', 'HLA-B*07:02', 'HLA-A*02:01', 'HLA-B*44:03', 'HLA-C*16:01', 'HLA-C*03:04', 'HLA-B*27:09', 'HLA-B*46:01', 'HLA-B*57:03', 'HLA-B*13:02', 'HLA-A*01:01', 'HLA-A*32:01', 'HLA-A*02:03', 'HLA-B*51:01', 'HLA-B*40:02', 'HLA-B*14:02', 'HLA-A*02:12', 'HLA-A*24:02', 'HLA-B*27:03', 'HLA-B*58:01', 'HLA-B*44:02', 'HLA-C*05:01', 'HLA-A*02:04', 'HLA-A*02:07', 'HLA-C*04:01', 'HLA-C*03:03', 'HLA-B*37:01', 'HLA-A*03:01', 'HLA-B*38:01', 'HLA-A*11:01', 'HLA-A*26:01', 'HLA-C*07:01', 'HLA-C*08:02', 'HLA-A*29:02', 'HLA-A*68:02', 'HLA-C*06:02', 'HLA-B*53:01', 'HLA-B*27:08', 'HLA-A*68:01', 'HLA-B*39:06', 'HLA-B*54:01', 'HLA-C*07:02', 'HLA-B*35:03', 'HLA-B*08:01', 'HLA-C*15:02', 'HLA-B*49:01', 'HLA-B*39:24', 'HLA-A*02:17', 'HLA-A*23:01', 'HLA-B*40:01', 'HLA-B*27:06', 'HLA-A*26:03', 'HLA-A*30:01', 'HLA-C*01:02', 'HLA-B*35:08', 'HLA-A*02:02', 'HLA-C*07:04', 'HLA-B*27:01', 'HLA-C*14:02', 'HLA-B*15:18', 'HLA-B*45:01', 'HLA-B*27:02', 'HLA-B*39:01', 'HLA-B*56:01', 'HLA-A*69:01', 'HLA-B*73:01', 'HLA-B*50:01', 'HLA-B*51:08', 'HLA-A*25:01', 'HLA-B*41:01', 'HLA-A*02:05', 'HLA-C*17:01', 'HLA-C*12:03', 'HLA-B*83:01', 'HLA-A*26:02', 'HLA-B*15:17', 'HLA-B*18:03', 'HLA-A*24:13', 'HLA-A*30:02', 'HLA-A*02:20', 'HLA-B*15:02', 'HLA-A*02:16', 'HLA-A*32:15', 'HLA-A*02:19', 'HLA-A*66:01', 'HLA-B*14:01', 'HLA-B*52:01', 'HLA-A*02:11', 'HLA-A*24:03', 'HLA-A*32:07', 'HLA-B*15:03', 'HLA-B*48:01', 'HLA-B*15:09', 'HLA-A*80:01', 'HLA-B*44:27', 'HLA-A*68:23', 'HLA-B*27:20']
MHC_TYPE_TO_IDX = {mhc_type: i for i, mhc_type in enumerate(MHC_TYPES)}
START_TOKEN = AA_TO_IDX['*']
MHC_SEQ_LEN = len(MHC_POSITION_MASK_IND) + 1
PEPTIDE_LEN = 15
EMB_SIZE = 32


def encode_peptide(peptide_str):
    return [START_TOKEN] + [AA_TO_IDX[peptide_str[i]] if i < len(peptide_str) else AA_TO_IDX['-']  for i in range(PEPTIDE_LEN - 1)]


def encode_mhc(mhc_str):
    return [START_TOKEN] + [AA_TO_IDX[mhc_str[i]] if i < len(mhc_str) else AA_TO_IDX['-'] for i in MHC_POSITION_MASK_IND]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
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
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(self.qw(x), self.kw(x), self.vw(x))[0]
        x = self.ln_2(x)
        x = x + self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
        return x


class CrossBlock(Block):
    def __init__(self, embed_dim=32, num_heads=4, *args, **kwargs):
        super(CrossBlock, self).__init__(embed_dim, num_heads, *args, **kwargs)
        self.ckw, self.cqw, self.cvw = nn.Linear(embed_dim, embed_dim), nn.Linear(embed_dim, embed_dim), nn.Linear(
            embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_3 = nn.LayerNorm(embed_dim)

    def forward(self, x, x_cross):
        x = self.ln_1(x)
        x = x + self.attn(self.qw(x), self.kw(x), self.vw(x))[0]
        x = self.ln_2(x)
        x = x + self.cross_attn(self.cqw(x), self.ckw(x_cross), self.cvw(x_cross))[0]
        x = self.ln_3(x)
        x = x + self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
        return x


class PanModel(nn.Module):
    def __init__(self, device='cpu'):
        super(PanModel, self).__init__()
        self.device = device
        self.pep = nn.Sequential(
            nn.Embedding(len(AMINO_ACIDS), EMB_SIZE),  # .from_pretrained(bl_embedding),
            PositionalEncoding(EMB_SIZE, max_len=PEPTIDE_LEN),
            nn.Tanh(),
        )
        self.pep_cross_blocks = nn.ModuleList(
            [CrossBlock(embed_dim=EMB_SIZE, num_heads=4) for i in range(3)]
        )
        self.mhc = nn.Sequential(
            nn.Embedding(len(AMINO_ACIDS), EMB_SIZE),
            PositionalEncoding(EMB_SIZE, max_len=MHC_SEQ_LEN),
            nn.Tanh(),
            *[Block(embed_dim=EMB_SIZE, num_heads=4) for i in range(3)],
        )
        self.head_1 = nn.Sequential(
            nn.Linear(EMB_SIZE, 128),
            # nn.Linear(PEPTIDE_LEN * 8, 64),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
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

    def forward_mhc(self, x_mhc):
        res = []
        for i in range(x_mhc.shape[0]):
            res.append(encode_mhc(MHC_TYPE_TO_IDX[MHC_TYPES[x_mhc[i]]]))
        res = torch.tensor(res).to(self.device)
        return self.mhc(res)

    def forward(self, x_pep, cmhc):
        x = self.pep(x_pep)
        for block in self.pep_cross_blocks:
            x = block(x, cmhc)
        return self.sig(
            self.head_2(
                x[:, 0, :].squeeze(1) + self.head_1(x[:, 0, :].squeeze(1))
            )
        )
