import torch
from utils import *
from metrics import MetronAtK
import random
import copy
from dataset import UserItemRatingDataset
from torch.utils.data import DataLoader
import numpy as np


class Engine(object):

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=self.config['top_k'])
        self.server_model_param = {}        
        self.client_model_params = {}       
        self.crit = torch.nn.BCELoss()

       
    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def _sample_participants(self):

        if self.config['clients_sample_ratio'] <= 1.0:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])

        return participants


    def fed_train_single_batch(self, model_client, batch_data, optimizers):

        raise NotImplementedError("This method has not been implemented yet.")
    
    def aggregate_clients_params(self, round_user_params):

        raise NotImplementedError("This method has not been implemented yet.")
     
    def fed_train_a_round(self, all_train_data, round_id):

        raise NotImplementedError("This method has not been implemented yet.")
       

    def final_fed_train_a_round(self, all_train_data, round_id):

        raise NotImplementedError("This method has not been implemented yet.")
        

    def fed_evaluate(self, evaluate_data):

        raise NotImplementedError("This method has not been implemented yet.")

    def _evaluate_core(self, evaluate_data, model_call_func):

        device = torch.device("cuda" if self.config['use_cuda'] else "cpu")
        test_users, test_items = evaluate_data[0].to(device), evaluate_data[1].to(device)
        negative_users, negative_items = evaluate_data[2].to(device), evaluate_data[3].to(device)

        num_users = self.config['num_users']
        test_scores = torch.empty(num_users, dtype=torch.float32, device=device)
        negative_score_list = []

        batch_size = 256 
        user_batches = [batch.to(device) for batch in torch.arange(num_users).split(batch_size)]
        

        with torch.no_grad():

            for batch_users_idx in user_batches:

                batch_users = test_users[batch_users_idx]  
                user_models = [self._get_user_model(user.item()).to(device) for user in batch_users]
                batch_test_items = test_items[batch_users_idx]
                test_scores[batch_users] = torch.stack([
                    model_call_func(model, user, items)
                    for model, user, items in zip(user_models, batch_users, batch_test_items)
                ])

                for idx, user in enumerate(batch_users):
                    user_mask = (negative_users == user)
                    neg_items = negative_items[user_mask]
                    neg_scores = model_call_func(user_models[idx], user.expand(neg_items.shape[0]), neg_items)
                    negative_score_list.append(neg_scores)  

        self._metron.subjects = [
            evaluate_data[0].numpy().tolist(),
            evaluate_data[1].numpy().tolist(),
            test_scores.cpu().numpy().tolist(),
            evaluate_data[2].numpy().tolist(),
            evaluate_data[3].numpy().tolist(),
            torch.cat(negative_score_list).cpu().numpy().tolist()
        ]

        return self._metron.cal_hit_ratio(), self._metron.cal_ndcg()


    def _model_call_single(self, model, user, items):

        if items.dim() == 0:  
            items = items.unsqueeze(0)  

        return model(items).squeeze()

    def _model_call_single_with_uid(self, model, user, items):
      
        if items.dim() == 0: 
            items = items.unsqueeze(0)  
            user = user.unsqueeze(0)
        return model(user, items).squeeze()

    def _model_call_multi(self, model, user, items):
      
        if items.dim() == 0:  
            items = items.unsqueeze(0)  
        return model(items, self.text_features, self.image_features).squeeze()   

    def _model_call_multi_with_uid(self, model, user, items):

        if items.dim() == 0:  
            items = items.unsqueeze(0)
            user = user.unsqueeze(0)
        return model(user, items, self.text_features, self.image_features).squeeze()

    def evaluate_single_modality(self, evaluate_data):

        return self._evaluate_core(evaluate_data, self._model_call_single)

    def evaluate_single_modality_with_uid(self, evaluate_data):

        return self._evaluate_core(evaluate_data, self._model_call_single_with_uid)

    def evaluate_multi_modality(self, evaluate_data):

        return self._evaluate_core(evaluate_data, self._model_call_multi)

    def evaluate_multi_modality_with_uid(self, evaluate_data):

        return self._evaluate_core(evaluate_data, self._model_call_multi_with_uid)
    
   