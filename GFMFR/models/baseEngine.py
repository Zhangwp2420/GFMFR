
import os

import numpy as np
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import torch
from utils import *
import random
from dataset import FusionDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from ourModel import GroupFusionModel,serverMLP
from sklearn.cluster import SpectralClustering
import random
from sklearn.metrics import pairwise_distances
import numpy as np



class BaseEngine:

    def __init__(self,config):

        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.server_model = GroupFusionModel(config).to(self.device)
        self.teacher_model = serverMLP(config).to(self.device)

        self.visual_features = torch.tensor(np.array(np.load("./dataset/" + self.config['dataset'] + "/image_features.npy")), dtype=torch.float32).to(self.device)
        self.text_features = torch.tensor(np.array(np.load("./dataset/" + self.config['dataset'] + "/text_features.npy")), dtype=torch.float32).to(self.device)
       
        self.ema_stage1 = {}  
        self.ema_stage2 = {}  
        self.ema_alpha = 0.9  
            
        self.mse = torch.nn.MSELoss()

        self.group_dict = {}
            
    def group_by_clients_params(self,round_participant_params,params_list):
    

        n_groups = self.config.get('group_num', 5)
        pca_variance = self.config.get('pca_variance', 0.95)
        use_pca = self.config.get('use_pca', True)  

        user_vectors = []
        user_ids = []

        for user in round_participant_params.keys():

            param_vectors = []
            for param_name in params_list:

                param_tensor = self.client_model_params[user][param_name]
                param_array = param_tensor.cpu().detach().numpy()
                flattened = param_array.flatten()  
                param_vectors.append(flattened)

            combined_vector = np.concatenate(param_vectors)
            user_vectors.append(combined_vector)
            user_ids.append(user)

        X = np.vstack(user_vectors)
     
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

  
        if use_pca and X_scaled.shape[1] > 1: 
            pca = PCA(n_components=pca_variance, random_state=42)
            X_processed = pca.fit_transform(X_scaled)
        else:
            X_processed = X_scaled

        if self.config['group_method'] == 'SpectralClustering':

            distances = pairwise_distances(X_processed, metric='euclidean')
            
            if np.all(distances == 0):
                median_dist = 1.0
            else:
                median_dist = np.median(distances[distances != 0])
            
            gamma = 1 / (2 * (median_dist ** 2))
            spectral = SpectralClustering(n_clusters=n_groups, affinity='rbf', gamma=gamma)
            labels = spectral.fit_predict(X_processed)

        elif self.config['group_method'] == 'KMeans':
            kmeans = KMeans(n_clusters=n_groups, random_state=42)
            labels = kmeans.fit_predict(X_processed)

        elif self.config['group_method'] == 'random':
            def random_grouping(X, n_groups):
             
                labels = np.random.randint(0, n_groups, size=X.shape[0])
                return labels

            def biased_random_grouping_last(X, n_groups):
                labels = np.zeros(X.shape[0], dtype=int)
                for i in range(X.shape[0]):
                    labels[i] = np.random.choice(
                        range(n_groups),
                        p=[0.5 / (n_groups - 1) if j != n_groups - 1 else 0.5 
                        for j in range(n_groups)]
                    )
                return labels
            
            labels = biased_random_grouping_last(X_processed, n_groups)
        else:
            raise ValueError(f"no grouping: {args.group_method}")


        if  self.group_dict :
            prev_group = self.group_dict.copy()  
        else:
            prev_group = {}

        self.group_dict = {} 

        for user_id, label in zip(user_ids, labels):
            self.group_dict.setdefault(label, []).append(user_id)
        
        self.group_dict = align_groups(prev_group,self.group_dict)  

        self.group_avg_items_embeddings = {}  
        for group_id, user_list in self.group_dict.items():
           
            embeddings = []
            for user_id in user_list:
                user_embed = self.client_model_params[user_id]['embedding_item.weight']
                embeddings.append(user_embed)
           
            stacked_embeddings = torch.stack(embeddings, dim=0)  
            avg_embeddings = torch.mean(stacked_embeddings, dim=0)  
            self.group_avg_items_embeddings[group_id] = avg_embeddings
            

    def create_quadruples(self):
        quadruples = []
        for group_id, avg_embedding_table in self.group_avg_items_embeddings.items():
            for item_idx in range(self.config['num_items']):
                avg_embedding = avg_embedding_table[item_idx]
                
                visual_feature = self.visual_features[item_idx]
                text_feature = self.text_features[item_idx]
                quadruples.append((group_id, visual_feature, text_feature, avg_embedding))

        group_ids = [q[0] for q in quadruples]
        visual_features = torch.stack([q[1] for q in quadruples], dim=0) 
        text_features = torch.stack([q[2] for q in quadruples], dim=0)   
        avg_embeddings = torch.stack([q[3] for q in quadruples], dim=0)  

        return group_ids,visual_features,text_features,avg_embeddings


    def train_group_fusion_module(self):

        fusedata = self.create_quadruples()
        Fusiondataset = FusionDataset(group_ids=fusedata[0],visual_features=fusedata[1]
                                ,text_features=fusedata[2],avg_item_embeddings=fusedata[3])
        fdataload =  DataLoader(Fusiondataset, batch_size=self.config['batch_size'], shuffle=True)

        self.server_model.train() 

        param_list = [
            {'params': self.server_model.embedding_group.parameters(), 'lr': self.config['lr']},
            {'params': self.server_model.fusion.parameters(), 'lr': self.config['lr']}
        ]

        if self.config['optimizer'] == 'sgd':
            server_optimizer = torch.optim.SGD(param_list,weight_decay=self.config['l2_regularization']) 
        else:
            server_optimizer = torch.optim.Adam(param_list,weight_decay=self.config['l2_regularization']) 
        
        stage1_loss = []


        for epoch in range(self.config['local_server_epoch']):
            batch_loss = 0.0
            all_num = 0

            for batch_id, batch_data in enumerate(fdataload):

                group_id, visual_feature, text_feature, avg_embedding = batch_data[0].to(self.device),batch_data[1].to(self.device),batch_data[2].to(self.device), batch_data[3].to(self.device)

                server_optimizer.zero_grad()

                yfuse = self.server_model(group_id, text_feature,visual_feature)
                loss =  self.mse(yfuse,avg_embedding) 
                loss.backward()
                server_optimizer.step()

                batch_loss = batch_loss + loss.item()
                all_num = all_num + len(group_id)

            epoch_loss = batch_loss / all_num
            stage1_loss.append(epoch_loss)

        self.unique_fusion_results = {}

        with torch.no_grad():  
            self.server_model.eval()  
       
            for group_id in self.group_avg_items_embeddings.keys():
               
                group_ids = torch.full((self.config['num_items'],), group_id, dtype=torch.long).to(self.device)
                fused_embeddings = self.server_model(group_ids,self.text_features, self.visual_features)
                self.unique_fusion_results[group_id] = fused_embeddings
     
        return stage1_loss

    def predict_group_preference(self):

        self.group_label = {}      
        with torch.no_grad():
            for group_id, client_ids in self.group_dict.items():
               
                if not client_ids:
                    continue
                    
              
                group_weight = None
                group_bias = None
                valid_clients = 0 
      
                for client_id in client_ids:
                    client_params = self.client_model_params[client_id]
                    if group_weight is None:
                        group_weight = torch.zeros_like(client_params['affine_output.weight'])
                        group_bias = torch.zeros_like(client_params['affine_output.bias'])
                        
                    group_weight += client_params['affine_output.weight']
                    group_bias += client_params['affine_output.bias']
                    valid_clients += 1
                    
                avg_weight = group_weight / valid_clients
                avg_bias = group_bias / valid_clients
                
                teacher_state = self.teacher_model.state_dict()
                teacher_state['affine_output.weight'].copy_(avg_weight)
                teacher_state['affine_output.bias'].copy_(avg_bias)
                self.teacher_model.load_state_dict(teacher_state)
                self.teacher_model.eval()
                    
                fused_embeddings = self.unique_fusion_results[group_id].to(self.device)
                preds = self.teacher_model(fused_embeddings)
                self.group_label[group_id] = preds

       
    def update_ema_losses(self, stage1_losses, stage2_losses):
        for client_id in stage1_losses.keys():
           
            if client_id not in self.ema_stage1:
                self.ema_stage1[client_id] = 1.0  
            else:
                self.ema_stage1[client_id] = (
                    self.ema_alpha * self.ema_stage1[client_id] 
                    + (1 - self.ema_alpha) * stage1_losses[client_id]
                ) 
         
            if client_id not in self.ema_stage2:
                self.ema_stage2[client_id] = 1.0
            else:
                self.ema_stage2[client_id] = (
                    self.ema_alpha * self.ema_stage2[client_id] 
                    + (1 - self.ema_alpha) * stage2_losses[client_id]
                )


    def calculate_alpha(self, round_id,  strategy="cosine_annealing"):

        total_rounds = self.config['num_round']
    
        if strategy == "linear_increase":
            return round_id / total_rounds  
        elif strategy == "exponential_increase":
            tau = 10  
            return 1 - np.exp(-round_id / tau)
        elif strategy == "cosine_annealing":
            return np.cos(np.pi * round_id / (2 * total_rounds))
        else:
            return 1.0  
        

    def compute_lambda_i(self, client_id, round_id):

        epsilon = 1e-8
     
        ema_s1 = self.ema_stage1.get(client_id, 1.0)
        ema_s2 = self.ema_stage2.get(client_id, 1.0)
        
   
        alpha = self.calculate_alpha(round_id, strategy="cosine_annealing")
       
        lambda_i = (ema_s1 / (ema_s2 + epsilon)) * alpha  
        return lambda_i

        
