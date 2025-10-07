import contextlib
import copy
import torch
import os
import random
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import *

import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from models.init import xavier_normal_initialization
from models.MR.modules import FusionLayer
from engine import Engine
import numpy as np
import torch.nn.init as init

class MLP(torch.nn.Module):

    def __init__(self, config):

        super(MLP, self).__init__()

        self.latent_size = config['latent_size']
        self.user_embedding = torch.nn.Parameter(torch.randn(1, self.latent_size))
        self.embedding_item = torch.nn.Embedding(num_embeddings=config['num_items'], embedding_dim=self.latent_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * self.latent_size, self.latent_size),  
            torch.nn.ReLU()
        )
        self.affine_output = torch.nn.Linear(in_features=self.latent_size, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.apply(xavier_normal_initialization)
        init.xavier_uniform_(self.user_embedding)

    def forward(self, user_indices, item_indices):

    
        batch_size = item_indices.shape[0]
        user_embed = self.user_embedding.expand(batch_size, -1)

        item_embed = self.embedding_item(item_indices)
   
        middle_result = self.mlp(torch.cat([user_embed, item_embed], dim=-1))
      
        pred = self.affine_output(middle_result)
        rating = self.logistic(pred)
        return rating


class FedNCFEngine(Engine):

    def __init__(self, config):
        super(FedNCFEngine, self).__init__(config)
    

        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.client_model = MLP(config).to(self.device)

        self.weight = {}  # store weight of each client 
        self._init_shared_parameters(self.client_model)


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
     
        params['embedding_item.weight'] = torch.nn.init.normal_(
            torch.empty_like(model.embedding_item.weight),
            std=0.01
        )
        self.initialized_params = params

        
    def fed_train_single_batch(self, model_client, batch_data, optimizer):
    
        users, items, ratings = batch_data[0].to(self.device), batch_data[1].to(self.device), batch_data[2].float().to(self.device)
        optimizer.zero_grad()
        ratings_pred = model_client(users,items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        return model_client, loss.item()


    def fed_train_a_round(self, all_train_data, round_id):
       
        participants =  self._sample_participants()
        # store users' model parameters of current round.
        round_participant_params = {}
        # store all the users' train loss and mae.
        all_loss = {}
        # perform model update for each participated user.
        for user in participants:
            loss = 0  
            model_client = MLP(self.config).to(self.device)

            if round_id != 0 and user in self.client_model_params:

                user_params = {}
                for name, param in model_client.state_dict().items():
                    if 'embedding_user.weight' in name:
                        user_params[name] = self.client_model_params[user][name].to(self.device)
                    else:
                        user_params[name] = self.server_model_param[name].to(self.device)
          
          
            if self.config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(model_client.parameters(),lr=self.config['lr'],weight_decay=self.config['l2_regularization'])
            else:
                optimizer = torch.optim.Adam(model_client.parameters(),lr=self.config['lr'],weight_decay=self.config['l2_regularization'])

        
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]

            self.weight[user] = len(user_train_data[0])

            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            sample_num = 0
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss_u = self.fed_train_single_batch(model_client, batch, optimizer)
                    loss += loss_u * len(batch[0])
                    sample_num += len(batch[0])
                all_loss[user] = loss / sample_num
      
            client_params = model_client.state_dict()
            self.client_model_params[user] = {k: v.cpu() for k, v in client_params.items()} 

            agg_params = {k: v for k, v in client_params.items() if  k != 'embedding_user.weight'}
            round_participant_params[user] = agg_params
            del model_client, optimizer, client_params
            torch.cuda.empty_cache()
        
        self.aggregate_clients_params(round_participant_params)
        return all_loss


    def aggregate_clients_params(self, round_user_params):
        
        with torch.no_grad():
            total_sum = sum(self.weight[user] for user in round_user_params.keys())
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
          
            stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
            
            with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
          
                for user, user_params in round_user_params.items():
              
                    w = self.weight[user] / total_sum
                    
          
                    for key in aggregated:
                        src_tensor = user_params[key]
                        
                   
                        if src_tensor.device != self.device:
                            src_tensor = src_tensor.to(self.device, non_blocking=True)
                        
                   
                        aggregated[key].add_(src_tensor.mul(w))
                
         
                for key in self.server_model_param:
                    self.server_model_param[key].data.copy_(aggregated[key])
            
        
            if stream:
                stream.synchronize()
        
    def final_fed_train_a_round(self, all_train_data, round_id):

        stage1_loss = self.fed_train_a_round(all_train_data, round_id)

        return stage1_loss,0

        
    def fed_evaluate(self, evaluate_data):

        return self.evaluate_single_modality_with_uid(evaluate_data)


    def _get_user_model(self, user_id):
        
        model = MLP(self.config).to(self.device) 
        
        
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

      
        for name, param in self.client_model.state_dict().items():
                if 'user_embedding' not in name:
                    params[name] = self.server_model_param[name].to(self.device)

        model.load_state_dict(params)
        return model.eval()
    
