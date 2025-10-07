
import torch
from torch.utils.data import  Dataset
import numpy as np

class FusionDataset(Dataset):
   
    def __init__(self, group_ids, visual_features, text_features, avg_item_embeddings):

        self.group_ids = torch.tensor(group_ids, dtype=torch.int64)                          
        self.visual_features = visual_features      
        self.text_features = text_features          
        self.avg_item_embeddings = avg_item_embeddings 

    def __getitem__(self, idx):
        return self.group_ids[idx], self.visual_features[idx], self.text_features[idx], self.avg_item_embeddings[idx]  

    def __len__(self):
        return len(self.group_ids)


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
    
class UserItemRatingPreferenceDataset(Dataset):
   
    def __init__(self, user_tensor, item_tensor, target_tensor,prefer_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.prefer_tensor = prefer_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], self.prefer_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
