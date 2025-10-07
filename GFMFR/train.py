import json
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import argparse
from data import SampleGenerator
from utils import *
import argparse
import torch



def get_training_config():
   
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--model', type=str, default='mmpfedrec')
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_num', type=int, default=0)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--lr_eta', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2_regularization', type=float, default=1e-4)
    
    
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--num_negative', type=int, default=4)
  
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=int, default=0)
    
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--local_server_epoch', type=int, default=50)
    parser.add_argument('--top_k', type=int, default=50)  
    parser.add_argument('--eval_setting', type=str, default='full') 
    parser.add_argument('--fusion_module', type=str, default='mlp')


    parser.add_argument('--loss_reg', type=str, default='all')

    parser.add_argument('--group_num', type=int, default=5)
    parser.add_argument('--pca_variance', type=float, default=0.95)
    parser.add_argument('--use_pca', action='store_true')
  
    parser.add_argument('--group_method', type=str, default='SpectralClustering')
  
    parser.add_argument('--initial_each_epoch', type=str, default='fromserver')
    

    parser.add_argument('--dataset_root_dir', type=str, default='dataset/')
    parser.add_argument('--inter_table_name', type=str, default='inter.csv')
    parser.add_argument('--dataset', type=str, default='KU-5core')
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--num_items', type=int)
    
    parser.add_argument('--logging_path', type=str, default='log/')

    parser.add_argument('--attack_method', type=str, default='None')
    parser.add_argument('--strength', type=float,default=0.0)
    parser.add_argument('--drop_modality', type=float,default=0.0)
    


    # Parse the arguments
    args = parser.parse_args()
    # Convert the arguments to a dictionary
    config = vars(args)

    return args,config


if __name__ == "__main__":
    
    _,config = get_training_config()
 
    torch.cuda.set_device(config['device_id'])
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logname = os.path.join(config['logging_path'], current_time + config['model'] + '-' + config['dataset'] + '-' + config['eval_setting']+'.txt')
    initLogging(logname)

    # DataLoader for training
    sample_generator = SampleGenerator(config=config)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data
    
    # get model
    enginer = get_trainer(config['model'],config)
    
    logging.info(f"Model Infor :{enginer}") 
    logging.info(json.dumps(config, indent=4))

    hit_ratio_list = []
    ndcg_list = []
    val_hr_list = []
    val_ndcg_list = []
    train_loss1_list = []
    train_loss2_list = []


    best_val_ndcg = 0
    best_val_hr = 0
    final_test_round_2 = 0  
    final_test_round_1 = 0  

    early_stop_patience = config.get('early_stop_patience', 10)

    no_improve_count = 0  

    for round_i in range(config['num_round']):
        logging.info('-' * 80)
        logging.info('Round {} starts !'.format(round_i))

        all_train_data = sample_generator.store_all_train_data(config['num_negative'])
        
        logging.info('-' * 80)
        logging.info('Training phase!')

        tr_loss1,tr_loss2 = enginer.final_fed_train_a_round(all_train_data=all_train_data, round_id=round_i)
 

        logging.info('-' * 80)
        logging.info('Validating phase!')
        val_hit_ratio, val_ndcg, v_loss = enginer.fed_evaluate(validate_data)
        logging.info(
            '[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round_i, val_hit_ratio, val_ndcg))

        if round_i > 0:

            ndcg_improved = val_ndcg > best_val_ndcg
            hr_improved = val_hit_ratio > best_val_hr 


            if ndcg_improved:
                best_val_ndcg = val_ndcg
                final_test_round_2 = round_i  

            if hr_improved:
                best_val_hr = val_hit_ratio
                final_test_round_1 = round_i 

            if  not (ndcg_improved or hr_improved):
                no_improve_count += 1
                logging.info(f'No improvement in both HR and NDCG for {no_improve_count} rounds.')
                if no_improve_count >= early_stop_patience:
                    logging.info(f'Early stopping triggered at round {round_i}!')
                    break  
            else:
                no_improve_count = 0  


        logging.info('-' * 80)
        logging.info('Testing phase!')
        hit_ratio, ndcg, te_loss = enginer.fed_evaluate(test_data)
        logging.info('[Testing Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(round_i, hit_ratio, ndcg))

        hit_ratio_list.append(hit_ratio)
        ndcg_list.append(ndcg)
        val_hr_list.append(val_hit_ratio)
        val_ndcg_list.append(val_ndcg)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    logging.info('hit_list: {}'.format(hit_ratio_list))
    logging.info('ndcg_list: {}'.format(ndcg_list))

    test_hr_based_on_hr = hit_ratio_list[final_test_round_1]
    test_ndcg_based_on_hr = ndcg_list[final_test_round_1]
    test_hr_based_on_ndcg = hit_ratio_list[final_test_round_2]
    test_ndcg_based_on_ndcg = ndcg_list[final_test_round_2]

    final_test_hr = (test_hr_based_on_hr + test_hr_based_on_ndcg) / 2
    final_test_ndcg = (test_ndcg_based_on_hr + test_ndcg_based_on_ndcg) / 2

    logging.info('Based on val_hr, best test HR: {:.4f}, NDCG: {:.4f} at round {}'.format(
        test_hr_based_on_hr, test_ndcg_based_on_hr, final_test_round_1))
    logging.info('Based on val_ndcg, best test HR: {:.4f}, NDCG: {:.4f} at round {}'.format(
        test_hr_based_on_ndcg, test_ndcg_based_on_ndcg, final_test_round_2))
    logging.info('Final averaged test HR: {:.4f}, Final averaged test NDCG: {:.4f}'.format(
        final_test_hr, final_test_ndcg))
