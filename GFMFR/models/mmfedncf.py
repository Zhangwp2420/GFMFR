import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import torch
from utils import *
from dataset import UserItemRatingDataset, UserItemRatingPreferenceDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from models.fedncf import FedNCFEngine,MLP
from models.baseEngine import BaseEngine

class mmfedncfEngine(FedNCFEngine, BaseEngine):

    def __init__(self, config):
     
        BaseEngine.__init__(self, config)
        FedNCFEngine.__init__(self, config)


        self._init_shared_parameters(MLP(config))

        self.weight = {}  # store weight of each client 
    
        self.ema_decay = 0.99
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
      
        params['embedding_item.weight'] = torch.nn.init.normal_(
            torch.empty_like(model.embedding_item.weight),
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


    def final_fed_train_a_round(self, all_train_data, round_id):
      
        if round_id == 0:
            
            stage_loss = self._base_stage_train(all_train_data, round_id)
        else:
         
            joint_loss = self._joint_stage_train(all_train_data, round_id)

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

        if round_id > 0:
            participants = self._get_valid_participants(participants)

        self.round_params = {}  

        all_loss = {}

        self.epoch_times = []
        
        for user in participants:
            
            model = self._init_model(user, round_id)

         
            user_train_data = [d[user] for d in data]
            self.weight[user] = len(user_train_data[0])

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
        agg_params = {k: v for k, v in user_params.items() if k != 'user_embedding'}

        round_params[user] = agg_params


    def _train_single_client(self, model, batch_data, optimizer, loss_fn, round_id):

      
        all_loss = {}

        for epoch in range(self.config['local_epoch']):
            loss = 0.0
            sample_num = 0
       
            for batch_id, batch in enumerate(batch_data):
                assert isinstance(batch[0], torch.LongTensor), "Batch data should contain LongTensor for user IDs."
             
                batch_users = batch[0].to(self.device)
                batch_items = batch[1].to(self.device)
                batch_ratings = batch[2].to(self.device)
                batch_group_prefs = batch[3].to(self.device) if len(batch) > 3 else None

                preds = model(batch_users, batch_items)
                
                if round_id == 0:
                    batch_loss = loss_fn(preds, batch_ratings.unsqueeze(1))
                else:
                    batch_loss = loss_fn(preds, batch_ratings.unsqueeze(1), batch_group_prefs,round_id)
               
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                loss += batch_loss.item() * len(batch_users)
                sample_num += len(batch_users)
          

            all_loss[epoch] = loss / sample_num

        return all_loss


    def _create_optimizer(self, model):
       
        param_list = [
            {'params': model.mlp.parameters(), 'lr': self.config['lr']},
            {'params': model.affine_output.parameters(), 'lr': self.config['lr']},
            {'params': [model.user_embedding], 'lr': self.config['lr']},
            {'params': model.embedding_item.parameters(), 'lr': self.config['lr']},
        ]

        if self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(param_list, weight_decay=self.config['l2_regularization'])  
        else:
            optimizer = torch.optim.Adam(param_list, weight_decay=self.config['l2_regularization'])    
        return optimizer
    

    def _init_model(self, user, round_id):

        model_client= MLP(self.config).to(self.device)  
        agg_params = {k: v for k, v in model_client.named_parameters() if k != 'user_embedding'}

  
        if round_id != 0 and user in self.client_model_params:
            user_params = {
                    k: v.to(self.device) 
                        for k, v in self.client_model_params[user].items()
                }
             
            user_params['embedding_item.weight'] = self.server_model_param['embedding_item.weight'].to(self.device)
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
            user_params['embedding_item.weight'] = self.initialized_params['embedding_item.weight'].to(self.device)
            user_params['affine_output.weight'] = self.initialized_params['affine_output.weight'].to(self.device)
            user_params['affine_output.bias'] = self.initialized_params['affine_output.bias'].to(self.device)
            user_params['mlp.0.weight'] = self.initialized_params['mlp.0.weight'].to(self.device)
            user_params['mlp.0.bias'] = self.initialized_params['mlp.0.bias'].to(self.device)

            model_client.load_state_dict(user_params)

        return model_client


    def _joint_loss(self, preds, ratings_tensor, group_prefs_tensor, round_id):
       
        cirt_loss = self.crit(preds, ratings_tensor)

        prob_dist_pred = torch.cat([preds, 1 - preds], dim=1)
        prefer_dist = torch.cat([group_prefs_tensor, 1 - group_prefs_tensor], dim=1)
        prefer_dist = torch.clamp(prefer_dist, min=1e-7, max=1.0)
     
        log_prob_dist_pred = torch.log(torch.clamp(prob_dist_pred, min=1e-7, max=1.0))
      
        kl_loss = F.kl_div(log_prob_dist_pred, prefer_dist, reduction='batchmean')

        if self.ema_ratio is None:
            
            self.ema_ratio = cirt_loss.item() / (kl_loss.item() + 1e-8)
        else:
          
            current_ratio = cirt_loss.item() / (kl_loss.item() + 1e-8)

            self.ema_ratio = self.ema_decay * self.ema_ratio + (1 - self.ema_decay) * current_ratio

        return cirt_loss + kl_loss * self.ema_ratio * self.calculate_alpha(round_id,"cosine_annealing")
    

    def _base_loss(self,preds, ratings_tensor):
      
        return self.crit(preds, ratings_tensor)

    def _update_group_info(self):

        group_params_list = ['mlp.0.weight','mlp.0.bias','affine_output.weight','affine_output.bias']
     
        self.group_by_clients_params(self.round_params,group_params_list)
        self.train_group_fusion_module()
        self.predict_group_preference()


    def _get_valid_participants(self, current_user):
        
        last_epoch_users = set(self.round_params.keys())
        return list(set(current_user) & last_epoch_users)

