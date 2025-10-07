import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from models.MR.modules import FusionLayer
import torch.nn.functional as F
import torch.nn as nn
from models.init import xavier_normal_initialization
import torch

'''
store function and class of our method ...

'''

class serverMLP(torch.nn.Module):

    def __init__(self, config):
        super(serverMLP, self).__init__()

        self.latent_size= config['latent_size']
        self.affine_output = torch.nn.Linear(in_features=self.latent_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_feature):
     
      
        logits = self.affine_output(item_feature)
        rating = self.logistic(logits)  # [0,1]
        return rating


class GroupFusionModel(torch.nn.Module):
    
    def __init__(self, config):
        super(GroupFusionModel, self).__init__()

        self.config = config
        self.latent_size = config['latent_size']
        self.embedding_size = config['embedding_size'] 
        self.num_gruops = config['group_num']  
        self.embedding_group = torch.nn.Embedding(num_embeddings=self.num_gruops, embedding_dim=self.embedding_size)
        self.fusion = FusionLayer(self.embedding_size, fusion_module=config['fusion_module'],
                                  latent_dim=self.latent_size)
    

        self.apply(xavier_normal_initialization)

    def forward(self, group_indices, txt_embed, vision_embed):

        group_embed = self.embedding_group(group_indices)     

        out = self.fusion(group_embed , txt_embed, vision_embed)

        return out
    
    def getGroupEmb(self,groupid):

        return self.embedding_group(torch.tensor(groupid, dtype=torch.long,device=next(self.parameters()).device))