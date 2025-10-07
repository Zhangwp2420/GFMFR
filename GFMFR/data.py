import torch
import random
import pandas as pd
from copy import deepcopy
import numpy as np


class SampleGenerator(object):
    """Construct dataset for NCF"""

    def __init__(self, config):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        dtype = {'userId': np.int32, 'itemId': np.int32, 'rating': np.float32}

        self.dataset_dir = f"{config['dataset_root_dir']}/{config['dataset']}/{config['inter_table_name']}"
        self.ratings = pd.read_csv(self.dataset_dir, sep=",", encoding="utf-8",dtype=dtype)
        self._validate_columns()
        self._preprocess(config.get('feedback_type', 'implicit'))

        self.user_pool = self.ratings['userId'].unique()
        self.item_pool = self.ratings['itemId'].unique()

        config['num_users'] = len(self.user_pool)
        config['num_items'] = len(self.item_pool)

        self.config = config
        self._precompute_negatives()
        self._split_datasets()
    
    def _validate_columns(self):
       
        required = ['userId', 'itemId', 'rating']
        missing = [col for col in required if col not in self.ratings.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")


    def _preprocess(self, feedback_type):
     
        ratings = self.ratings
        if feedback_type == 'explicit':
          
            max_rating = ratings['rating'].max()
            ratings['rating'] = ratings['rating'] / max_rating
        else:  # implicit
           
            ratings['rating'] = np.where(ratings['rating'] > 0, 1.0, 0.0)

        self.preprocess_ratings = ratings


  

    def _precompute_negatives(self):
       
        user_items = self.ratings.groupby('userId')['itemId'].apply(set)
        all_items = set(self.item_pool)
        
   
        self.negatives_all = {
            uid: list(all_items - pos_items)
            for uid, pos_items in user_items.items()
        }
        
     
        self.negatives_sample = {}

        for uid in self.negatives_all:
            avail_neg = self.negatives_all[uid]
            if len(avail_neg) >= 198:
        
                sample = np.random.choice(avail_neg, 198, replace=False)
            else:
         
                sample = np.random.choice(avail_neg, 198, replace=True)

            self.negatives_sample[uid] = sample


    def _split_datasets(self):

    
        processed_ratings = self.preprocess_ratings
  
        grouped = processed_ratings.groupby('userId', sort=False)['timestamp']
        ranks = grouped.rank(method='first', ascending=False).values
        
 
        test_mask = ranks == 1
        val_mask = ranks == 2
        train_mask = ranks > 2
        
      
        self.test_ratings = self._mask_to_df(processed_ratings, test_mask)
        self.val_ratings = self._mask_to_df(processed_ratings, val_mask)
        self.train_ratings = self._mask_to_df(processed_ratings, train_mask)

    def _mask_to_df(self, source_df, mask):
     
        return source_df[mask][['userId', 'itemId', 'rating']].reset_index(drop=True)

    def store_all_train_data(self, num_negatives):

     
        users = []
        items = []
        ratings = []

        for uid in range(self.config['num_users']):
         
            pos_data = self.train_ratings[self.train_ratings['userId'] == uid]
            pos_items = pos_data['itemId'].values
            pos_ratings = pos_data['rating'].values
            

            num_pos = len(pos_items)
            num_needed = num_negatives * num_pos

            avail_negatives = self.negatives_all[uid]
          
            if len(avail_negatives) >= num_needed:
               
                neg_items = avail_negatives[:num_needed]
            else:
              
                neg_items = np.random.choice(
                    avail_negatives, 
                    size=num_needed,
                    replace=True
                )           
          
            user_arr = np.repeat(uid, num_pos + num_needed)
            item_arr = np.concatenate([pos_items, neg_items])
            rating_arr = np.concatenate([pos_ratings, np.zeros(num_needed, dtype=np.float32)])
            
            users.append(user_arr)
            items.append(item_arr)
            ratings.append(rating_arr)
        
        
        return [users,items,ratings]
  

    @property
    def validate_data(self):
       
        return self._generate_eval_data(self.val_ratings,"val")

    @property
    def test_data(self):
       
        return self._generate_eval_data(self.test_ratings,"test")

  

    def _generate_eval_data(self, df, stage):
       
        users = df['userId'].values.astype(np.int32)
        items = df['itemId'].values.astype(np.int32)
        
        if self.config.get('eval_setting') == 'full':
        
            neg_users = []
            neg_items = []
            for uid in users:
                user_negs = self.negatives_all[uid]
                neg_users.extend([uid] * len(user_negs))
                neg_items.extend(user_negs)
        else:
          
            neg_per_user = 99
            neg_users = np.repeat(users, neg_per_user)
            
         
            if stage == "val":
                neg_items = np.concatenate(
                    [self.negatives_sample[uid][:neg_per_user] for uid in users]
                )
            else:
                neg_items = np.concatenate(
                    [self.negatives_sample[uid][neg_per_user:] for uid in users]
                )
        
        return (
            torch.from_numpy(users),
            torch.from_numpy(items),
            torch.from_numpy(np.array(neg_users, dtype=np.int32)),
            torch.from_numpy(np.array(neg_items, dtype=np.int32))
        )