
import torch
import logging
import importlib
import random
import numpy as np
import torch



def get_trainer(alias,config):
  
    if not alias:
        raise ValueError("Alias must be provided.")

    model_name = alias.lower()
    module_path = '.'.join(['models', model_name])

    try:
 
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            raise ImportError(f"Module '{module_path}' not found.")
       
        model_module = importlib.import_module(module_path)
        trainer_class = getattr(model_module, f'{alias}Engine')
        trainer_instance = trainer_class(config=config)
        return trainer_instance
    except ImportError as e:
        logging.error(f"Failed to import module '{module_path}': {e}")
        raise
    except AttributeError:
        logging.error(f"Trainer class '{alias}Engine' not found in module '{module_path}'.")
        raise

def get_trainer_with_id(alias,config):

    if not alias:
        raise ValueError("Alias must be provided.")

    model_name = alias.lower()
    module_path = '.'.join(['modelsWid', model_name])

    try:
      
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            raise ImportError(f"Module '{module_path}' not found.")
      
        model_module = importlib.import_module(module_path)
       
        trainer_class = getattr(model_module, f'{alias}Engine')
        trainer_instance = trainer_class(config=config)
        return trainer_instance
    except ImportError as e:
        logging.error(f"Failed to import module '{module_path}': {e}")
        raise
    except AttributeError:
        logging.error(f"Trainer class '{alias}Engine' not found in module '{module_path}'.")
        raise

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_args(args):  
    """Logs all the arguments and their values."""
    for arg in vars(args):
        logging.info(f"Argument {arg}: {getattr(args, arg)}")



def reorder_group_dict(dict1):
    
    min_values_with_keys = [(min(dict1[key]), key) for key in dict1]
    sorted_keys = [key for _, key in sorted(min_values_with_keys)]
    new_group_dict = {new_key: dict1[old_key] for new_key, old_key in enumerate(sorted_keys)}
    return new_group_dict


from scipy.optimize import linear_sum_assignment
import numpy as np


def align_groups(prev_groups, current_groups):

    if not prev_groups :
        return current_groups 
  
    prev_sets = {k: set(v) for k, v in prev_groups.items()}
    current_sets = {k: set(v) for k, v in current_groups.items()}
    
    prev_keys = sorted(prev_sets.keys())
    current_keys = sorted(current_sets.keys())
    
    cost_matrix = -np.array([[len(current_sets[c] & prev_sets[p]) 
                           for p in prev_keys] 
                          for c in current_keys])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {}
    used_prev_keys = set()
    
    for i, c_idx in enumerate(row_ind):
        p_key = prev_keys[col_ind[i]]
        c_key = current_keys[c_idx]
        mapping[c_key] = p_key
        used_prev_keys.add(p_key)
    
    all_current = sorted(current_sets.keys())
    matched = set(mapping.keys())

    for c_key in all_current:
        if c_key not in matched:

            new_key = 0
            while new_key in mapping.values():
                new_key += 1
            mapping[c_key] = new_key
    

    aligned_groups = {}
    for c_key in all_current:
        aligned_groups[mapping[c_key]] = current_groups[c_key]

    aligned_groups = dict(sorted(aligned_groups.items()))
    return aligned_groups