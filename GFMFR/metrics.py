import math
import pandas as pd
import numpy as np

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        test_users, test_items, test_scores = subjects[0], subjects[1], subjects[2]
        neg_users, neg_items, neg_scores = subjects[3], subjects[4], subjects[5]
       
        user_to_item = dict(zip(test_users, test_items))
        
    
        full = pd.DataFrame({
            'user': neg_users + test_users,
            'item': neg_items + test_items,
            'score': neg_scores + test_scores
        })
        
      
        full['test_item'] = full['user'].map(user_to_item)
        full['is_test_item'] = full['item'] == full['test_item']
        
      
        full.sort_values(['user', 'score'], ascending=[True, False], kind='stable', inplace=True)
       
        full['rank'] = full.groupby('user').cumcount() + 1
        
    
        self._subjects = full[full['is_test_item']][['user', 'rank']].copy()


    def cal_hit_ratio(self):
        """Hit Ratio @ top_K"""

        top_k = self._top_k
        hit_count = (self._subjects['rank'] <= top_k).sum()
        return hit_count / len(self._subjects)
      

    def cal_ndcg(self):
        top_k = self._top_k
        mask = self._subjects['rank'] <= top_k
        ndcg_values = np.log2(2) / np.log2(1 + self._subjects.loc[mask, 'rank'])
        return ndcg_values.sum() / len(self._subjects)
