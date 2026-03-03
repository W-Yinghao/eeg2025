import torch
import torch.nn as nn
from .SSSM import SSSM
from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        # self.backbone = CBraMod(
        #     in_dim=200, out_dim=200, d_model=200,
        #     dim_feedforward=800, seq_len=30,
        #     n_layer=12, nhead=8
        # )
        # if param.use_pretrained_weights:
        #     map_location = torch.device(f'cuda:{param.cuda}')
        #     self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone = SSSM(
            in_channels=200, res_channels=200,
            skip_channels=200, out_channels=200,
            num_res_layers=param.n_layer,
            diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200,
            diffusion_step_embed_dim_out=200,
            s4_lmax=570,
            s4_d_state=64,
            s4_dropout=param.dropout,
            s4_bidirectional=True,
            s4_layernorm=True,
            codebook_size_t=param.codebook_size_t,
            codebook_size_f=param.codebook_size_f,
            if_codebook=False)
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            self.backbone.load_state_dict(new_state_dict)
        self.backbone.proj_out = nn.Sequential()
        # self.avgpooling = nn.AdaptiveAvgPool2d(1)
        # self.attn = nn.MultiheadAttention(200, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(16*10*200, 10*200),
            nn.ELU(),
            nn.Linear(10*200, 200),
            nn.ELU(),
            nn.Linear(200, 1)
        )

    def forward(self, x, train = False):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num*seq_len*200)
        # feats = self.avgpooling(feats).contiguous().view(bz, 200)
        out = self.classifier(feats)
        out = out.contiguous().view(bz)
        return out

