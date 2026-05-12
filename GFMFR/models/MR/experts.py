import torch
import torch.nn as nn


class SumExpert(nn.Module):


    def __init__(self):
        super(SumExpert, self).__init__()

    def forward(self, x, y, z):
       
        return x + y + z


class MLPExpert(nn.Module):
 

    def __init__(self, embed_size):
        super(MLPExpert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, embed_size)
        )

    def forward(self, x, y, z):
      
        c = torch.cat([x, y, z], dim=-1)  
        c = self.mlp(c)

        return c


class MultiHeadAttentionExpert(nn.Module):
   

  
    def __init__(self, embed_size, num_heads=4):
        super(MultiHeadAttentionExpert, self).__init__()

        in_dim = embed_size * 3
        self.attn = nn.MultiheadAttention(in_dim, num_heads)
        self.fc_out = nn.Linear(in_dim, embed_size)

    def forward(self, x, y, z):
        c = torch.cat([x, y, z], dim=-1) 

        out, weights = self.attn(c, c, c)

        out = self.fc_out(out) 

        return out


class GateExpert(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super(GateExpert, self).__init__()
        self.txt_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Sigmoid()
        )
        self.vis_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Sigmoid()
        )
        self.id_gate = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Sigmoid()
        )

        self.fusion = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, idf, txt, vis):
     
        id_values = self.id_gate(idf)
        txt_values = self.txt_gate(txt)
        vis_values = self.vis_gate(vis)
        
        
 
        feat = torch.cat([id_values, txt_values, vis_values], dim=1)

        output = self.fusion(feat)

        return output
