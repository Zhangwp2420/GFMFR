import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.init import xavier_normal_initialization
from models.MR.experts import SumExpert, MLPExpert, MultiHeadAttentionExpert, GateExpert


class GatingNetwork(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0., latent_dim=128):
        super(GatingNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        weights = F.softmax(out, dim=1)

     
        if len(weights.size()) > 1:
            weights = torch.mean(weights, dim=0)

        return weights.unsqueeze(0)


class SwitchingFusionModule(nn.Module):

    def __init__(self, in_dim, embed_dim, dropout=0., latent_dim=128):
        super(SwitchingFusionModule, self).__init__()

        self.idx = -1

        self.router = GatingNetwork(in_dim * 3, 3, dropout, latent_dim)

        self.experts = nn.ModuleList([
            SumExpert(),
            MLPExpert(embed_dim),
            GateExpert(embed_dim, embed_dim)
        ])

    def forward(self, x, y, z):
      
        outs = [expert(x, y, z) for expert in self.experts]


        c = torch.cat(outs, dim=1)

        scores = self.router(c)

        weights = torch.softmax(scores, dim=1)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[0, i] * outs[i]

        return out


class FusionLayer(nn.Module):

    def __init__(self, in_dim, fusion_module='moe', latent_dim=128):
        super(FusionLayer, self).__init__()

        self.id_affine = nn.Linear(in_dim, latent_dim)
        self.txt_affine = nn.Linear(in_dim, latent_dim)
        self.vis_affine = nn.Linear(in_dim, latent_dim)

        if fusion_module == 'moe':
            self.fusion = SwitchingFusionModule(latent_dim, latent_dim, dropout=0.5, latent_dim=latent_dim)
        elif fusion_module == 'sum':
            self.fusion = SumExpert()
        elif fusion_module == 'mlp':
            self.fusion = MLPExpert(latent_dim)
        elif fusion_module == 'attention':
           # self.fusion = MultiHeadAttentionExpert(latent_dim, 8)
           self.fusion = MultiHeadAttentionExpert(latent_dim, 1)
        elif fusion_module == 'gate':
            self.fusion = GateExpert(latent_dim, latent_dim)
        else:
            raise ValueError('Invalid fusion module, currently only support: moe, sum, mlp, and attention.')

    def forward(self, id_feat, txt_feat, vis_feat):

   
        id_feat = self.id_affine(id_feat)
        txt_feat = self.txt_affine(txt_feat)
        vis_feat = self.vis_affine(vis_feat)
        return self.fusion(id_feat, txt_feat, vis_feat)

