import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from models.MR.modules import FusionLayer
from models.init import xavier_normal_initialization
from engine import Engine
import contextlib
from utils import *
from dataset import (
    UserItemRatingDataset,
    UserItemRatingPreferenceDataset,
    FusionDataset
)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    SpectralClustering
)
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

class FedRAP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_size = config['latent_size']

        self.embedding_user = nn.Embedding(config['num_users'], self.latent_size)
        self.item_personality = nn.Embedding(config['num_items'], self.latent_size)
        self.item_commonality = nn.Embedding(config['num_items'], self.latent_size)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.latent_size, self.latent_size),
            nn.ReLU()
        )

        self.affine_output = nn.Linear(self.latent_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, users, items):
        u = self.embedding_user(users)
        ip = self.item_personality(items)
        ic = self.item_commonality(items)

        x = self.mlp(torch.cat([u, ip + ic], dim=-1))
        logits = self.affine_output(x)
        pred = self.sigmoid(logits)

        return pred, ip, ic




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
    
   

class GFMFREngine(Engine):

    def __init__(self,config):
        
        
        Engine.__init__(self, config)
        self.config = config
        self.device = torch.device("cuda" if config.get('use_cuda', False) else "cpu")

        self.client_model = FedRAP(config).to(self.device)
        self.lr_network = config['lr']
        self.lr_args = config['lr'] * config['num_items']

        self.crit = nn.BCELoss()
        self.independency = nn.MSELoss()
        self.reg = nn.L1Loss()
        self.mse = torch.nn.MSELoss()

        self.alpha = config.get("alpha", 0.1)
        self.beta = config.get("beta", 0.1)

        self.client_model_params = {}
        self.server_model_param = {}


        self.server_model = GroupFusionModel(config).to(self.device)
        self.teacher_model = serverMLP(config).to(self.device)

        self.visual_features = torch.tensor(np.array(np.load("./dataset/" + self.config['dataset'] + "/image_features.npy")), dtype=torch.float32).to(self.device)
        self.text_features = torch.tensor(np.array(np.load("./dataset/" + self.config['dataset'] + "/text_features.npy")), dtype=torch.float32).to(self.device)
        self.group_dict = {}
        self.history = []

        self._init_shared_parameters(self.client_model)
        self.ema_decay = 0.90
        self.ema_ratio = None  


    def _init_shared_parameters(self, model):
    
        params = {}
        mlp_layer = model.mlp[0]  
        params['mlp.0.weight'] = torch.nn.init.kaiming_normal_(
            torch.empty_like(mlp_layer.weight),
            nonlinearity='relu'
        )

        params['mlp.0.bias'] = torch.zeros_like(mlp_layer.bias)
        params['affine_output.weight'] = torch.nn.init.xavier_uniform_(
            torch.empty_like(model.affine_output.weight),   
        )
        params['affine_output.bias'] = torch.zeros_like(model.affine_output.bias)
        params['item_commonality.weight'] = torch.nn.init.normal_(
            torch.empty_like(model.item_commonality.weight),
            std=0.01
        )
        self.initialized_params = params

    def instance_user_train_loader(self, user_data, round_id):
       
        if round_id == 0:
            dataset = UserItemRatingDataset(
                torch.LongTensor(user_data[0]),
                torch.LongTensor(user_data[1]),
                torch.FloatTensor(user_data[2])
            )
        else:
            dataset = UserItemRatingPreferenceDataset(
                torch.LongTensor(user_data[0]),
                torch.LongTensor(user_data[1]),
                torch.FloatTensor(user_data[2]),
                user_data[3]
            )
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)


    def _update_hyperparams(self, *args, **kwargs):

        iteration = args[0]
        self.lr_args *= self.config.get('decay_rate',0.9)
        self.lr_network *= self.config.get('decay_rate',0.8)
        self.alpha = math.tanh(iteration / 10) * self.alpha
        self.beta = math.tanh(iteration / 10) * self.beta

    def final_fed_train_a_round(self, all_train_data, round_id):
       
        if round_id == 0:
            stage_loss = self._base_stage_train(all_train_data, round_id)
        else:
            joint_loss = self._joint_stage_train(all_train_data, round_id)

        self._update_hyperparams(round_id)
        self._update_group_info() 

        return 0,0

    def _base_stage_train(self, all_train_data, round_id):
   
        return self._run_client_training(
            data=all_train_data,
            round_id=round_id,
            loss_fn=self._base_loss,
        )


    def _joint_stage_train(self, all_train_data, round_id):

  
        full_train_data = self.integrate_group_preferences(all_train_data)

        return self._run_client_training(
            data=full_train_data,
            round_id=round_id,
            loss_fn=self._joint_loss,
        )


    def _run_client_training(self, data, round_id, loss_fn):
    
        participants =  self._sample_participants()

        self.round_params = {}  

        all_loss = {}

        for user in participants:
        
            model = self._init_model(user, round_id)

   
            user_train_data = [d[user] for d in data]
            batch_data = self.instance_user_train_loader(user_train_data, round_id)
          
            optimizer = self._create_optimizer(model)

            user_loss = self._train_single_client(
                model, batch_data, optimizer, 
                loss_fn, round_id
            )
            
            self._record_params(user, model, self.round_params)
            all_loss[user] = user_loss

        self.aggregate_clients_params(self.round_params)

        return all_loss


    def integrate_group_preferences(self, all_train_data):
     
     
        user_train_data = [[], [], [], []]
       
        users, items, ratings = all_train_data

        for i in range(len(users)):
            user_id = users[i][0]  
            group_id = None
            for g_id, user_list in self.group_dict.items():
                    if user_id in user_list:
                        group_id = g_id
                        break
            if group_id is None:
                user_train_data[3].append(ratings[i])
            else: 
                group_preferences = self.group_label[group_id]
                user_train_data[3].append(group_preferences[items[i]])  
            
            user_train_data[0].append(users[i]) 
            user_train_data[1].append(items[i])  
            user_train_data[2].append(ratings[i]) 
            
        return user_train_data
    

    def _record_params(self, user, model, round_params):
   
        user_params = {
            k: v.cpu().detach() 
                for k, v in model.state_dict().items()
        }
        self.client_model_params[user] = user_params
        agg_params = {k: v for k, v in user_params.items() if k != 'embedding_user.weight' and k != 'item_personality.weight'}
        round_params[user] = agg_params


    def _train_single_client(self, model, batch_data, optimizer, loss_fn, round_id):
     
        all_loss = {}
        for epoch in range(self.config['local_epoch']):
            loss = 0.0
            sample_num = 0
            for batch_id, batch in enumerate(batch_data):
                try:
                  
                    assert isinstance(batch[0], torch.LongTensor), "Batch data should contain LongTensor for user IDs."
                    batch_users = batch[0].to(self.device)
                    batch_items = batch[1].to(self.device)
                    batch_ratings = batch[2].to(self.device)
                    batch_group_prefs = batch[3].to(self.device) if len(batch) > 3 else None

                 
                    preds,item_personality, item_commonality = model(batch_users, batch_items)

                    if round_id == 0:
                        batch_loss = loss_fn(preds, batch_ratings.unsqueeze(1),item_personality, item_commonality)
                    else:
                        batch_loss = loss_fn(preds, batch_ratings.unsqueeze(1), batch_group_prefs, round_id,item_personality, item_commonality)

                
                    if torch.isnan(batch_loss):
                        print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_id}. Skipping this batch.")
                        continue 

               
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    loss += batch_loss.item() * len(batch_users)
                    sample_num += len(batch_users)

                except Exception as e:
                    print(f"Error during training at epoch {epoch}, batch {batch_id}: {e}")
                    continue  
         
            if sample_num > 0:
                all_loss[epoch] = loss / sample_num
            else:
                all_loss[epoch] = float('nan')  

        return all_loss


    def _create_optimizer(self, model):
      
        param_list = [
                {'params': model.affine_output.parameters(), 'lr': self.lr_network},
                {'params': model.mlp.parameters(), 'lr': self.lr_network},
                {'params': model.item_personality.parameters(), 'lr': self.lr_args},
                {'params': model.item_commonality.parameters(), 'lr': self.lr_args},
                {'params': model.embedding_user.parameters(), 'lr': self.lr_network}
            ]
        if self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(param_list, weight_decay=self.config['l2_regularization'])  
            
        else:
            optimizer = torch.optim.Adam(param_list, weight_decay=self.config['l2_regularization'])
           
        return optimizer
    

    def _init_model(self, user, round_id):

        model_client= FedRAP(self.config).to(self.device)

        if round_id != 0 and user in self.client_model_params:
            user_params = {
                    k: v.to(self.device) 
                        for k, v in self.client_model_params[user].items()
                }
             
            user_params['item_commonality.weight'] = self.server_model_param['item_commonality.weight'].to(self.device)
            if self.config['initial_each_epoch'] == 'fromserver':
                user_params['mlp.0.weight'] = self.server_model_param['mlp.0.weight'].to(self.device)
                user_params['mlp.0.bias'] = self.server_model_param['mlp.0.bias'].to(self.device)
                user_params['affine_output.weight'] = self.server_model_param['affine_output.weight'].to(self.device)
                user_params['affine_output.bias'] = self.server_model_param['affine_output.bias'].to(self.device)

            model_client.load_state_dict(user_params)
        else:
            user_params = {
                    k: v.to(self.device) 
                        for k, v in model_client.state_dict().items()
                }
            
            user_params['item_commonality.weight'] = self.initialized_params['item_commonality.weight'].to(self.device)
            user_params['affine_output.weight'] = self.initialized_params['affine_output.weight'].to(self.device)
            user_params['affine_output.bias'] = self.initialized_params['affine_output.bias'].to(self.device)
            user_params['mlp.0.weight'] = self.initialized_params['mlp.0.weight'].to(self.device)
            user_params['mlp.0.bias'] = self.initialized_params['mlp.0.bias'].to(self.device)
            
            model_client.load_state_dict(user_params)

        return model_client


    def _joint_loss(self, preds, ratings_tensor, group_prefs_tensor,round_id,*args, **kwargs):
       
        item_personality, item_commonality = args[0], args[1]

        crit_loss = self._base_loss(preds, ratings_tensor,item_personality, item_commonality)

        prob_dist_pred = torch.cat([preds, 1 - preds], dim=1)
        prefer_dist = torch.cat([group_prefs_tensor, 1 - group_prefs_tensor], dim=1)
        prefer_dist = torch.clamp(prefer_dist, min=1e-7, max=1.0)
        log_prob_dist_pred = torch.log(torch.clamp(prob_dist_pred, min=1e-7, max=1.0))
      
        kl_loss = F.kl_div(log_prob_dist_pred, prefer_dist, reduction='batchmean')

        if torch.isnan(crit_loss) :
            print(f'crit_Loss is nan at epoch: {round_id}. Exiting.')
            return crit_loss

        if  torch.isnan(kl_loss):
            print(f'kl_Loss is nan at epoch: {round_id}. Exiting.')
            return kl_loss

        if self.ema_ratio is None:

            self.ema_ratio = crit_loss.item() / (kl_loss.item() + 1e-8)
            
        else:
           
            current_ratio = crit_loss.item() / (kl_loss.item() + 1e-8)

            self.ema_ratio = self.ema_decay * self.ema_ratio + (1 - self.ema_decay) * current_ratio

        alpha_t = self.calculate_alpha(round_id,"cosine_annealing")

        return crit_loss + kl_loss * self.ema_ratio * alpha_t
    
    def _base_loss(self, preds, ratings_tensor,*args, **kwargs):
       
        item_personality, item_commonality = args[0], args[1]
        return self.calculate_loss(preds, ratings_tensor,item_personality, item_commonality)
     
    def calculate_loss(self, interaction, *args, **kwargs):

        pred, truth, item_personality, item_commonality = interaction, args[0], args[1], args[2]

        dummy_target = torch.zeros_like(item_commonality).to(self.device)

        loss = self.crit(pred, truth) \
               - self.alpha * self.independency(item_personality, item_commonality) \
               + self.beta * self.reg(item_commonality, dummy_target)

        return loss
    
    def _update_group_info(self):

        group_params_list = ['mlp.0.weight','mlp.0.bias','affine_output.weight','affine_output.bias']
        self.group_by_clients_params(self.round_params,group_params_list)
        self.train_group_fusion_module()  
        self.predict_group_preference()   


    def group_by_clients_params(self, round_participant_params, params_list):
     
        n_groups = self.config.get('group_num', 5)
        pca_variance = self.config.get('pca_variance', 0.95) 
        use_pca = self.config.get('use_pca', True)  

        user_vectors = []
        user_ids = []

        for user in round_participant_params.keys():

            param_vectors = []
            for param_name in params_list:
             
                if param_name not in self.client_model_params[user]:
                    raise KeyError(f" {param_name} not in {user}")

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

        else:
            raise ValueError(f"no group method")

        if  self.group_dict :
            prev_group = self.group_dict.copy() 
        else:
            prev_group = {}

        self.group_dict = {}

        for user_id, label in zip(user_ids, labels):
            self.group_dict.setdefault(label, []).append(user_id)
        
        self.group_dict = align_groups(prev_group,self.group_dict)  

        self.history.append(self.group_dict.copy())  

        self.group_avg_items_embeddings = {}  
        for group_id, user_list in self.group_dict.items():

            embeddings = []
            for user_id in user_list:
                user_embed = self.client_model_params[user_id]['item_commonality.weight']
                embeddings.append(user_embed)

            stacked_embeddings = torch.stack(embeddings, dim=0)  
            avg_embeddings = torch.mean(stacked_embeddings, dim=0)  
            self.group_avg_items_embeddings[group_id] = avg_embeddings


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

    def aggregate_clients_params(self, round_user_params):

        with torch.no_grad():
            first_user_params = next(iter(round_user_params.values()))
            if not hasattr(self, 'server_model_param') or not self.server_model_param:
                self.server_model_param = {
                    k: torch.zeros_like(v, device=self.device)
                    for k, v in first_user_params.items()
                }
            aggregated = {
                k: torch.zeros_like(v, device=self.device)
                for k, v in first_user_params.items()
            }
            stream = torch.cuda.Stream(device=self.device) if self.device .type == 'cuda' else None
            with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
                total_users = len(round_user_params)
                for user_params in round_user_params.values():
                    for key in aggregated:
                        src_tensor = user_params[key]
                        if src_tensor.device != self.device :
                            src_tensor = src_tensor.to(self.device , non_blocking=True)
                        aggregated[key].add_(src_tensor)
                for key in aggregated:
                    self.server_model_param[key].data.copy_(
                        aggregated[key] / total_users
                    )
            if stream:
                stream.synchronize()   

    def fed_evaluate(self, evaluate_data):

        return self.evaluate_single_modality_with_uid(evaluate_data)
    
    def _model_call_single_with_uid(self, model, user, items):
 
        if items.dim() == 0: 
            items = items.unsqueeze(0) 
            user = user.unsqueeze(0)

        return model(user, items)[0].squeeze()

    def _get_user_model(self, user_id):
        
        model = FedRAP(self.config).to(self.device)  
        
        if user_id in self.client_model_params:
            params = {
                k: v.to(self.device) 
                for k, v in self.client_model_params[user_id].items()
            }
        else:
            params = {
                k: v.to(self.device) 
                for k, v in self.client_model.state_dict().items()
            }

        for k, v in self.server_model_param.items():
            params[k] = v.to(self.device)

        model.load_state_dict(params)
        return model.eval()
